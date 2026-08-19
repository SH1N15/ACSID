# ACSID on AMD — MI300X 192GB single-GPU branch

This folder is the AMD variant of the project: **same ACSID method** (collaborative
signal fused into the RQ-VAE input via `P` + adaptive `alpha_i`), different
training configuration than the A10 plan. Design & experiment matrix:
[`PLAN_AMD.md`](PLAN_AMD.md); base plan: [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md).

Key differences vs the NVIDIA/A10 path:

| | A10 24GB (`main`) | MI300X 192GB (this branch) |
|---|---|---|
| fine-tune | QLoRA 4-bit (planned) | **full-param bf16** |
| optimizer | paged_adamw_32bit (bnb) | **adamw_torch** |
| GRPO group size | 2-4 | **16 (paper config)** |
| RL data | 10-30% subset | **100%** |
| launcher | torchrun/accelerate | plain `python`, single GPU |

## What's in here

```
acsid_amd/
├── PLAN_AMD.md          AMD variant of the project plan (v4-amd)
├── sft.py               AMD SFT entrypoint (upstream sft.py minus bitsandbytes,
│                        plus path-injection + fixes, see "Fixes" below)
├── rl.py                AMD GRPO entrypoint (optim=adamw_torch, ReReTrainer kept)
├── sft.sh               single-GPU SFT launch (self-locating)
├── rl.sh                single-GPU GRPO launch (self-locating)
├── run_experiments.sh   full 10-run matrix (6 SFT + 4 GRPO)
├── config/zero2_opt.yaml  BACKUP single-GPU config (not wired; DeepSpeed unused)
└── requirements.txt     ROCm requirements (torch installed separately)
```

Phase 2 (SID construction: Item2Vec → fusion+P → RQ-VAE → 3 SID sets → CSVs)
is **shared** with the A10 branch — run `acsid/generate_sid.py` unchanged
(`--device cuda:0` works on ROCm; PyTorch maps the CUDA API onto HIP).

## Fixes applied on this branch (vs first draft)

1. **Import paths** — `sft.py`/`rl.py` live outside `MiniOneRec/` but import
   `data`, `minionerec_trainer`, `sasrec` from it. Added a `sys.path`
   injection at the top of both so they resolve regardless of cwd.
2. **`freeze_LLM` NameError** — upstream references `original_vocab_size`
   without defining it; now captured before `resize_token_embeddings`.
   (Latent: runs use `freeze_LLM False`, but this branch shouldn't ship a crash.)
3. **TokenExtender now honors the real `--sid_index_path`** — upstream always
   loaded `<dataset>.index.json` (the shipped baseline), silently ignoring
   `index.text.json` / `index.fixed.json` / `index.adaptive.json`. Without this,
   per-mode runs could tokenize SID strings with missing tokens.
4. **Shell scripts are self-locating** — they `cd` into `MiniOneRec/` and run
   `python ../acsid_amd/{sft,rl}.py`, fixing the old cwd mismatch
   (`./data/...` needs MiniOneRec cwd; `python sft.py` needed acsid_amd cwd).
5. **`config/zero2_opt.yaml`** — was stale `DEEPSPEED`/`num_processes: 8`;
   now a clearly-marked single-GPU backup, unused by the scripts.

## Running on the cloud MI300X

### 0. Environment (ROCm) — isolated, single GPU

**Process**: build a dedicated venv (`bash acsid_amd/setup_env.sh`), which installs
PyTorch for ROCm + the requirements WITHOUT touching any CUDA/NVIDIA env on the
machine. torch is deliberately NOT in `requirements.txt` so it can't accidentally
fall back to a CUDA/CPU wheel — the setup script installs it first from the ROCm
index.

Default install location: `${PROJECT_ROOT}/.venv-amd` (gitignored).

```bash
# clone + check out this branch
git clone https://github.com/SH1N15/ACSID.git
cd ACSID && git checkout acsid-amd

# single isolated venv (idempotent; re-run to upgrade/repair)
bash acsid_amd/setup_env.sh
source .venv-amd/bin/activate

# sanity: ROCm sees the GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True AMD Instinct MI300X ...
```

Knobs (env vars) for the setup script:

| var | default | meaning |
|---|---|---|
| `VENV_DIR` | `${PROJECT_ROOT}/.venv-amd` | location of the isolated virtualenv |
| `ROCM_TAG` | `rocm6.2` | PyTorch ROCm wheel tag (use `rocm6.3.4`, etc. to match your driver) |
| `PY_VER` | `python3.10` | base interpreter used to create the venv |

> **conda alternative**: `mamba env create -f acsid_amd/environment.yml` →
> `conda activate acsid-amd` → then the same two pip steps the venv path runs
> (torch from the ROCm index, then `pip install -r
> acsid_amd/requirements.txt`). Use only if your stack prefers conda.

### 1. Phase 2 — SID construction (shared pipeline, runs on MI300X)

```bash
bash experiments/run_phase2_sid.sh Industrial_and_Scientific
# or stepwise from MiniOneRec/rq/ — see ../acsid/README.md
```

Produces `./data/Amazon/index/<dataset>.index.{text,fixed,adaptive}.json` and
`./data/Amazon/{text,fixed,adaptive}/...`, which everything below consumes.

### 2. SFT baseline smoke (Phase 1, official SIDs, single run)

```bash
BASE_MODEL=/abs/path/to/Qwen2.5-3B-Base bash acsid_amd/sft.sh
# (sft.sh points at index.text.json by default; for the OFFICIAL-SID baseline
#  swap --sid_index_path to ./data/Amazon/index/Industrial_and_Scientific.index.json
#  and the CSVs to the upstream ./data/Amazon/{train,valid}/*.csv)
```

### 3. Full 10-run matrix (Phase 3+4)

```bash
BASE_MODEL=/abs/path/to/Qwen2.5-3B-Base bash acsid_amd/run_experiments.sh
```

## Known AMD caveats

- **bitsandbytes is gone** — anything importing it will crash; this branch's
  scripts no longer reference it. `optim="adamw_torch"` everywhere.
- **SDP attention** runs on the math backend (`enable_flash_sdp(False)` in
  `rl.py`): correct but slower than CUDA flash-attn. If generation is the
  bottleneck, ROCm's `flash-attn` fork is the upgrade path.
- **triton==3.2.0** in requirements may need the ROCm triton build on some
  stacks; harmless unless something imports it (nothing here does directly).
- **`triton`/`deepspeed` are kept** as inert dependencies (deepspeed for the
  multi-GPU backup config only).
