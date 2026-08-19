# ACSID — Phase 2: Adaptive Collaborative Semantic ID construction

Injects collaborative-filtering (Item2Vec) signals into the **RQ-VAE input**
stage via a learnable projection `P` and per-item adaptive weight `alpha_i`,
instead of at the RL reward stage. The downstream SFT / GRPO / constrained
decoding are untouched — only the SID construction changes. See
`PROJECT_PLAN.md` for the full design.

---

## What this phase builds

Three SID variants share one training protocol (same codebook=256×3, e_dim,
layers, AdamW, epochs, seed) and differ only in the RQ-VAE input:

| mode     | input to RQ-VAE                                   | P trained? |
|----------|---------------------------------------------------|------------|
| `text`   | `L2norm(z_text)`                                  | no         |
| `fixed`  | `L2norm[(1-α)·L2norm(z_text) + α·L2norm(P(z_cf))]`, α=`alpha_max` constant | yes |
| `adaptive` | same, but `α_i = alpha_max·min(1, log(1+n_i)/log(1+n_ref))`; cold-start items α=0 → pure text | yes |

`n_i` = item interaction count over **train only** (no leakage, plan §12.1);
`n_ref` = median of nonzero item frequencies; `alpha_max=0.3`.

## New files (`acsid/`, sibling of `MiniOneRec/`)

```
acsid/
├── __init__.py
├── item2vec.py            # Word2Vec SGNS over train sequences -> cf.npy [N,256]
├── adaptive_fusion.py     # compute_item_freq / compute_alpha / FusionModule(P + L2norm sum)
├── generate_sid.py        # end-to-end orchestrator (run on cloud A10)
├── regenerate_csv_sid.py  # rewrite SID columns of existing CSVs + info/*.txt (no .inter)
└── analyze_collision.py   # collision rate / unique ratio across the 3 variants
experiments/
├── run_phase2_sid.sh      # one-shot runner (cd MiniOneRec/rq, run pipeline + analysis)
└── results/               # collision.json lands here
```

## Modified files (back-compat: `--mode text` == stock behavior)

- `MiniOneRec/rq/datasets.py` — adds `FusedEmbDataset` (text+cf+alpha); `EmbDataset` kept.
- `MiniOneRec/rq/rqvae.py` — `--mode/--cf_path/--alpha_path/--alpha_max`; builds `FusionModule` when `mode!=text`; passes it to `Trainer`.
- `MiniOneRec/rq/trainer.py` — optimizer includes P; new `_prepare_input` fuses per-batch; train/valid use `z_i`; checkpoint stores `fusion_state_dict`.
- `MiniOneRec/rq/generate_indices.py` — argparse; rebuilds `FusionModule` from `ckpt["args"]` + `fusion_state_dict`; fuses the full matrix; original sinkhorn de-collision loop + token prefix/JSON contract preserved.
- `MiniOneRec/requirements.txt` — `gensim`, `peft`.

## Data flow

```
text .npy  ──┐
cf.npy     ──┤─► FusedEmbDataset ─► FusionModule ─► z_i ─► RQ-VAE ─► ckpt(+P)
alpha.npy  ──┘                                              │
                                                            ▼
                                            generate_indices (rebuild P, fuse, get_indices,
                                                              sinkhorn de-collision)
                                                            │
                                                            ▼
                                            <dataset>.index.<mode>.json
                                                            │
                                            regenerate_csv_sid ─► data/Amazon/<mode>/{train,valid,test,info}
```

---

## Running on the cloud A10

All runs execute from `MiniOneRec/rq/` (same cwd convention as `rqvae.sh`).
**Do not run locally on Windows** — there is no GPU and no python env here.

### 0. Install deps

```bash
cd MiniOneRec
pip install -r requirements.txt
# verify
python -c "import gensim, torch; print('ok', torch.cuda.is_available())"
```

### 1. Smoke test (text mode only, short epochs — confirm the line wiring)

```bash
cd MiniOneRec/rq
python ../../acsid/generate_sid.py \
    --dataset Industrial_and_Scientific \
    --epochs 200 --batch_size 2048 --eval_step 50 \
    --device cuda:0 --modes text
# should produce ../data/Amazon/index/Industrial_and_Scientific.index.text.json
# and ../data/Amazon/text/{train,valid,test,info}/...
python ../../acsid/analyze_collision.py \
    --base ../data/Amazon --dataset Industrial_and_Scientific \
    --include_upstream --out_json ../../experiments/results/collision.json
```

### 2. Full Phase 2 (all three modes, full protocol)

```bash
# one-shot (tune epochs/batch in the script if needed)
bash experiments/run_phase2_sid.sh Industrial_and_Scientific
```

or equivalently:

```bash
cd MiniOneRec/rq
python ../../acsid/generate_sid.py \
    --dataset Industrial_and_Scientific \
    --epochs 10000 --batch_size 20480 --eval_step 50 \
    --device cuda:0 --modes text fixed adaptive
python ../../acsid/analyze_collision.py \
    --base ../data/Amazon --dataset Industrial_and_Scientific \
    --include_upstream --out_json ../../experiments/results/collision.json
```

### What gets produced

```
MiniOneRec/data/Amazon/index/
  cf.npy                                    # Item2Vec [N,256] (train-only)
  alpha.npy                                 # per-item adaptive α
  Industrial_and_Scientific.index.text.json
  Industrial_and_Scientific.index.fixed.json
  Industrial_and_Scientific.index.adaptive.json
MiniOneRec/rq/output/<dataset>/<mode>/<timestamp>/best_collision_model.pth   # ×3
MiniOneRec/data/Amazon/{text,fixed,adaptive}/{train,valid,test,info}/...      # ×3
experiments/results/collision.json
```

The shipped upstream `index.json` and CSVs are **not overwritten** (outputs go to `.index.<mode>.json` and `data/Amazon/<mode>/`).

---

## Acceptance / interpretation

`analyze_collision.py` prints a table and a directional check; the plan §3.5
expects **adaptive ≤ fixed ≤ text** on collision rate. If the trend is wrong,
inspect (in order):

1. `cf.npy` coverage — how many items got a vector (`items_with_vector` printed by item2vec).
2. `alpha.npy` stats — is `mean<<alpha_max` and `max==alpha_max`? (mostly cold-start → adaptive≈text is fine, but not the goal.)
3. RQ-VAE training loss / collision — does `fixed/adaptive` reach at least the text collision rate? If loss diverges, see § 两步预训练 below.

### If P joint-training is unstable (plan §12.5 fallback)

Switch to two-step: (1) freeze RQ-VAE, pretrain `P` to reconstruct the text
embedding (MSE from `P(z_cf)` toward `z_text`) for a few hundred steps; (2)
unfreeze RQ-VAE and continue joint training. This is a code change to
`trainer.py` (add a `--pretrain_p_steps` flag), not yet implemented — open if
needed.

---

## API notes for later phases

- **SFT/GRPO** only need the per-mode CSV + `index.<mode>.json` + `info/*.txt`
  under `data/Amazon/<mode>/`. Point `sft.sh`/`rl.sh` at those paths; the
  training code itself is unchanged.
- The new LLM token vocabulary per mode comes from the mode's `index.<mode>.json`
  (via `sft.py` `TokenExtender`); each mode resizes embeddings independently.
- `generate_indices.py` is now argparse-driven: `--ckpt_path / --output_file
  / --device`. To regenerate SIDs from a different checkpoint without touching
  the orchestrator, call it directly from `MiniOneRec/rq/`.
