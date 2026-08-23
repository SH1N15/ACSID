# vendored upstream MiniOneRec — DO NOT EDIT ABOVE THIS LINE IN UPSTREAM PATHS WITHOUT KNOWING
# the provenance; the original repo was inlined flat (its .git removed) so rq/*.py changes
# inside MiniOneRec/ are tracked here alongside the ACSID contributions.

ACS-ID built on top of MiniOneRec (arXiv:2510.24431).

# ACSID — Adaptive Collaborative Semantic ID construction

ACSID injects collaborative-filtering (Item2Vec) signals into the **RQ-VAE
input** stage via a learnable projection `P` and a per-item adaptive weight
`alpha_i`, rather than at the RL reward stage. The downstream SFT / GRPO /
constrained decoding are untouched — only the SID construction changes.

**Research question**: should collaborative signal enter at SID construction,
or be learned later by RL reward? Answer from the experiments: SID-stage
injection wins at every stage, and RL only partially substitutes for it.

## Results (Amazon Industrial_and_Scientific, seed 42, CC = 0 everywhere)

All three objectives proven:

**1. ACSID reduces SID collision** (`experiments/results/collision.json`):

| | upstream | text | fixed (α=0.3) | adaptive (ours) |
|---|---|---|---|---|
| collision rate | 0.0043 | 0.0030 | **0.0** | 0.0011 |
| unique SIDs / 3686 | 3670 | 3675 | 3686 | 3682 |

**2. ACSID improves SFT recommendation** and **3. ACSID+GRPO beats Text+GRPO**
(`MiniOneRec/results/eval_*.json`):

| metric | SFT text | SFT adaptive | GRPO text | GRPO adaptive |
|---|---|---|---|---|
| NDCG@1 | 0.0547 | 0.0613 (+12%) | 0.0695 | **0.0763 (+9.8%)** |
| NDCG@5 | 0.0667 | 0.0778 (+17%) | 0.0913 | **0.0969 (+6.1%)** |
| HR@10 | 0.0984 | 0.1160 (+18%) | 0.1357 | **0.1425 (+5.0%)** |
| NDCG@50 | 0.0911 | 0.1031 (+13%) | 0.1126 | **0.1222 (+8.5%)** |

Headline finding: GRPO lifts both SID variants hugely (text NDCG@5 +36.9%,
adaptive +24.6%) but the adaptive-over-text lead only shrinks from +16.6% to
+6.1% — RL's ranking reward *partially substitutes* collaborative injection
yet never closes the gap. Popularity-stratified analysis
(`acsid/phase5_analysis.py`, common subset n=4258) shows the substitution is
bucket-local: GRPO multiplies the weak text variant most where it was weakest
(low-popularity bucket ×6.2 vs adaptive's ×2.8).

**Case study** — text-SID collisions are almost exclusively same-title
product-variant pairs (same product, different sizes). Adaptive separates
**11/11** of them, and does so at the *leaf* SID level while sharing the first
two levels: e.g. items 7/8 ("UltraSource 192033 Hamburger Patty Paper") share
`<a_198><b_110>` and split into `<c_117>` / `<c_54>` — the CF residual
disambiguates variants *inside* their semantic neighborhood.

**Ablation nuance (honest)**: fixed (constant α=0.3) beats adaptive at SFT
(NDCG@10 0.0952 vs 0.0850). Stratification localizes the gap to
mid/low-popularity items where adaptive's log-schedule under-displaces
(mean α 0.22–0.29 vs 0.3); the cold-start hypothesis is refuted (cold items
are bit-identical in both modes because `P(bias=False)` maps `z_cf=0` to 0).
Adaptive keeps the safer story (α→0 for cold items) at a measurable mid-tail
cost; tuning α_max upward is the obvious follow-up.

## Method

Residual injection (v2) — z_text is never normalized; P learns direction only;
α bounds the displacement to ≤ α_max·‖z_text‖:

```
α_i     = α_max · min(1, log(1+n_i) / log(1+n_ref))     # n_i: train interactions
z_i     = z_text + α_i · ‖z_text‖ · Normalize(P(z_cf))  # P: Linear(256→2560), jointly
                                                        # trained with the RQ-VAE
text    : z_i = z_text (byte-identical to upstream MiniOneRec)
fixed   : α_i = 0.3 for every item
adaptive: α_i per formula; cold-start items (n_i=0) degenerate to pure text
```

Design docs: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (A10 variant),
[`acsid_amd/PLAN_AMD.md`](acsid_amd/PLAN_AMD.md) (this branch's plan),
[`acsid/README.md`](acsid/README.md) (Phase 2 architecture).

## Repo layout

```
acsid/              SID construction core (shared by both hardware branches):
                    item2vec · adaptive_fusion · generate_sid · regenerate_csv_sid
                    · analyze_collision · phase5_analysis
acsid_amd/          MI300X branch: sft.py / rl.py (full-param bf16, adamw_torch),
                    sft.sh / rl.sh / run_experiments.sh (PHASES/SEEDS/SKIP_MODES),
                    setup_env.sh, PLAN_AMD.md
MiniOneRec/         vendored upstream (arXiv:2510.24431) with ACSID edits in rq/*.py;
                    evaluate.py / calc.py / minionerec_trainer.py used as-is
experiments/        run scripts + results/ (collision.json, phase5_analysis.json)
MiniOneRec/results/ per-sample eval JSONs (SFT ×3 modes + GRPO ×2 modes)
HANDOFF.md          operational handoff: environment pitfalls, run history,
                    cross-session resume recipes (the real ops manual)
```

## Reproduction (1× AMD MI300X 192GB, ROCm 7.2.3, torch 2.11, Python 3.12)

```bash
# 0. environment (idempotent; builds a --system-site-packages venv around the
#    preinstalled ROCm torch — see setup_env.sh header for why torch is NOT in
#    requirements.txt)
bash acsid_amd/setup_env.sh && source .venv-amd/bin/activate

# 1. SID construction: Item2Vec (CPU) -> alpha -> 3× RQ-VAE -> 3 index tables
#    -> SID columns rewritten into per-mode CSVs (results: collision.json)
bash experiments/run_phase2_sid.sh

# 2. SFT + eval (text/fixed/adaptive × seed 42, ~3-4h each on MI300X)
cd MiniOneRec
SKIP_MODES="" PHASES="sft"   SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh
SKIP_MODES="" PHASES="eval"  SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh

# 3. GRPO (text/adaptive, 1 epoch, full RL data; ~10-17h/mode — spans multiple
#    8h cloud sessions; rerunning the same command auto-resumes from the last
#    checkpoint, finished modes are skipped)
SKIP_MODES="" PHASES="grpo"  SEEDS_STR="42" bash ../acsid_amd/run_experiments.sh
#    then evaluate the GRPO checkpoints (see HANDOFF.md §3 for the exact loop)

# 4. stratified analysis + collision case study (CPU-only, runs anywhere)
python ../acsid/phase5_analysis.py
```

Known deviations from the plan (recorded in HANDOFF.md): GRPO runs 1 epoch
instead of 2 (8h session cap; full RL data kept), single seed (42) throughout
(100GB storage quota), fixed stays SFT-only by design (ablation control).

## Upstream provenance

`MiniOneRec/` is vendored **flat** (its own `.git` removed on 2026-08-19) from:

- origin: <https://github.com/AkaliKong/MiniOneRec.git>
- commit: `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`
- branch: `main` (2026-05-14)

Local edits on top of that commit live in `MiniOneRec/rq/` (`datasets.py`,
`rqvae.py`, `trainer.py`, `generate_indices.py`) and `MiniOneRec/requirements.txt`.
Run `git log -- MiniOneRec/rq/` to see them.

## Status

All planned experiments complete (Phase 2 SID construction, Phase 3 SFT+eval,
Phase 4 GRPO+eval, Phase 5 stratified analysis + case study). `main` **is**
the executed MI300X path (merged from `acsid-amd`, which stays as the
development branch); the superseded A10/QLoRA-era plan is preserved under the
`a10-archive` tag. Operational history and every environment pitfall:
[`HANDOFF.md`](HANDOFF.md).
