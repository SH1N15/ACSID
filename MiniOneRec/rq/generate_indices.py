"""Generate index.json (item -> SID tokens) from a trained RQ-VAE checkpoint.

ACSID extension: when the checkpoint was trained with mode != text, the
FusionModule (P + L2-normalized adaptive sum) is rebuilt from the
checkpoint's args/state and applied to reconstruct the SAME fused embeddings
used during training, so the generated SIDs reflect the collaborative-text
representation rather than raw text. The sinkhorn collision-breaking loop
and the token-prefix / JSON contract are unchanged from the upstream.
"""

import collections
import json
import logging
import os
import sys

import numpy as np
import torch
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import EmbDataset, FusedEmbDataset
from models.rqvae import RQVAE

# Allow `from acsid.adaptive_fusion import FusionModule` regardless of cwd.
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS))        # .../MiniOneRec
_PROJECT_ROOT = os.path.dirname(_REPO_ROOT)                 # .../DewAlgo
for _p in (_PROJECT_ROOT, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from acsid.adaptive_fusion import FusionModule  # noqa: E402


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def build_fused_matrix(dataset, fusion, mode, alpha_max, device, batch=4096):
    """Materialize the [N, D_text] fused embedding matrix used for indexing.

    Replicates the trainer's _prepare_input exactly:
      text     -> L2norm(z_text)
      fixed    -> fusion(z_text, z_cf, alpha=alpha_max constant)
      adaptive -> fusion(z_text, z_cf, alpha=alpha.npy per item)
    """
    N = len(dataset)
    D = dataset.dim
    out = torch.empty(N, D, dtype=torch.float32)
    fusion_dev = fusion.to(device) if fusion is not None else None
    with torch.no_grad():
        if fusion_dev is not None:
            fusion_dev.eval()
        for s in range(0, N, batch):
            e = min(s + batch, N)
            z_text = torch.from_numpy(dataset.text[s:e]).to(device)
            if fusion_dev is not None:
                z_cf = torch.from_numpy(dataset.cf[s:e]).to(device)
                if mode == "fixed":
                    alpha = torch.full((e - s,), float(alpha_max),
                                       device=device, dtype=z_text.dtype)
                else:  # adaptive
                    alpha = torch.from_numpy(dataset.alpha[s:e]).to(device).view(-1)
                z_i = fusion_dev(z_text, z_cf, alpha)
            else:
                z_i = torch.nn.functional.normalize(z_text, p=2, dim=-1)
            out[s:e] = z_i.cpu()
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Generate index.json from an RQ-VAE checkpoint (ACSID-aware)")
    ap.add_argument("--ckpt_path", required=True, help="path to best_collision_model.pth")
    ap.add_argument("--output_file", required=True, help="output index.json path")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64, help="batch size for index generation")
    ap.add_argument("--fuse_batch", type=int, default=4096, help="batch size for fusing the full matrix")
    args = ap.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(
        args.ckpt_path,
        map_location=torch.device('cpu'),
        weights_only=False
    )

    ck_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    fusion_state_dict = ckpt.get("fusion_state_dict")
    mode = getattr(ck_args, "mode", "text")
    alpha_max = getattr(ck_args, "alpha_max", 0.3)

    # Rebuild the SAME dataset the trainer used (text + cf + alpha), so both
    # the RQ-VAE dims and the fusion sources come back identically.
    cf_path = getattr(ck_args, "cf_path", "") or None
    alpha_path = getattr(ck_args, "alpha_path", "") or None
    dataset = FusedEmbDataset(
        text_path=ck_args.data_path,
        cf_path=cf_path,
        alpha_path=alpha_path,
    )

    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=ck_args.num_emb_list,
        e_dim=ck_args.e_dim,
        layers=ck_args.layers,
        dropout_prob=ck_args.dropout_prob,
        bn=ck_args.bn,
        loss_type=ck_args.loss_type,
        quant_loss_weight=ck_args.quant_loss_weight,
        beta=ck_args.beta,
        kmeans_init=ck_args.kmeans_init,
        kmeans_iters=ck_args.kmeans_iters,
        sk_epsilons=ck_args.sk_epsilons,
        sk_iters=ck_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    # Rebuild fusion if this checkpoint was trained with CF injection.
    fusion = None
    if mode != "text":
        if fusion_state_dict is None:
            raise RuntimeError(f"checkpoint mode={mode} but no fusion_state_dict found")
        fusion = FusionModule(cf_dim=dataset.cf_dim, text_dim=dataset.dim)
        fusion.load_state_dict(fusion_state_dict)
        print(f"[ACSID] rebuilt fusion: mode={mode} cf_dim={dataset.cf_dim} text_dim={dataset.dim} alpha_max={alpha_max}")

    # Materialize the fused matrix once; indexing/de-collision index into it.
    fused = build_fused_matrix(dataset, fusion, mode, alpha_max, device, batch=args.fuse_batch)
    N = fused.size(0)
    print(f"[ACSID] fused matrix shape: {tuple(fused.shape)}")

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for s in tqdm(range(0, N, args.batch_size), desc="Indexing"):
        e = min(s + args.batch_size, N)
        d = fused[s:e].to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    # There are often duplicate items in the dataset, and we no longer differentiate them
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = fused[collision_items].to(device)

            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)) or ".", exist_ok=True)
    with open(args.output_file, 'w') as fp:
        json.dump(all_indices_dict, fp)


if __name__ == "__main__":
    main()
