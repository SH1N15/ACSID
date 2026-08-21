import argparse
import os
import random
import sys
import torch
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader

from datasets import EmbDataset, FusedEmbDataset
from models.rqvae import RQVAE
from trainer import Trainer

# Allow `from acsid.adaptive_fusion import FusionModule` regardless of the
# cwd. acsid/ lives one level above the MiniOneRec repo root.
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS))        # .../MiniOneRec
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)                 # .../DewAlgo
for _p in (_PROJECT_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from acsid.adaptive_fusion import FusionModule  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Games/Games.emb-llama-td.npy",
                        help="Input text embedding path (.npy).")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    # ---- ACSID: collaborative-text fusion at the RQ-VAE input ----
    parser.add_argument('--mode', type=str, default='text', choices=['text', 'fixed', 'adaptive'],
                        help='text = no fusion (pure L2norm(text)); fixed = alpha=alpha_max constant; '
                             'adaptive = alpha_i from --alpha_path')
    parser.add_argument('--cf_path', type=str, default='',
                        help='path to cf.npy [N, cf_dim] from Item2Vec; required when mode != text')
    parser.add_argument('--alpha_path', type=str, default='',
                        help='path to alpha.npy [N] (per-item adaptive weight); '
                             'used when mode=adaptive; ignored for fixed/text')
    parser.add_argument('--alpha_max', type=float, default=0.3, help='upper bound of the CF weight')
    # cf_dim is inferred from cf.npy (FusedEmbDataset.cf_dim); kept only for logging/parity.
    parser.add_argument('--cf_dim', type=int, default=256, help='expected Item2Vec dim (read from cf.npy if present)')

    return parser.parse_args()


if __name__ == "__main__":
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset

    All three modes go through FusedEmbDataset so the data pipeline stays
    uniform. cf_path/alpha_path are read only when provided; in text mode
    no FusionModule is built and the trainer returns raw z_text (residual
    injection: z = z_text + alpha * ||z_text|| * Normalize(P(z_cf)),
    z_text never normalized).
    """
    cf_path = args.cf_path if args.cf_path else None
    alpha_path = args.alpha_path if args.alpha_path else None
    data = FusedEmbDataset(text_path=args.data_path, cf_path=cf_path, alpha_path=alpha_path)

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    # P is created when the collaborative branch can actually contribute.
    # In text mode alpha=0 and P would be unconstrained (gradient-free),
    # so we skip it; the trainer fuses with fusion=None => L2norm(text).
    fusion = None
    if args.mode != 'text':
        fusion = FusionModule(cf_dim=data.cf_dim, text_dim=data.dim)
        print(f"[ACSID] mode={args.mode} cf_dim={data.cf_dim} text_dim={data.dim} alpha_max={args.alpha_max}")
    else:
        print(f"[ACSID] mode=text (no P; input = L2norm(text))")

    print(model)
    data_loader = DataLoader(data, num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args, model, len(data_loader), fusion=fusion)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss", best_loss)
    print("Best Collision Rate", best_collision_rate)

