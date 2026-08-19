# vendored upstream MiniOneRec — DO NOT EDIT ABOVE THIS LINE IN UPSTREAM PATHS WITHOUT KNOWING
# the provenance; the original repo was inlined flat (its .git removed) so rq/*.py changes
# inside MiniOneRec/ are tracked here alongside the ACSID contributions.

ACS-ID built on top of MiniOneRec (arXiv:2510.24431).

# ACSID — Adaptive Collaborative Semantic ID construction

ACSID injects collaborative-filtering (Item2Vec) signals into the **RQ-VAE
input** stage via a learnable projection `P` and a per-item adaptive weight
`alpha_i`, rather than at the RL reward stage. The downstream SFT / GRPO /
constrained decoding are untouched — only the SID construction changes.

- Full design & experiment plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (v4).
- Phase 2 implementation guide & cloud run: [`acsid/README.md`](acsid/README.md).
- One-shot runner: [`experiments/run_phase2_sid.sh`](experiments/run_phase2_sid.sh).

## Repo layout

```
acsid/              ACSID new modules (Item2Vec, fusion, orchestrator, regenerate, analyze)
experiments/        run scripts + results/ (collision.json lands here)
MiniOneRec/         vendored upstream MiniOneRec, with ACSID edits in rq/*.py
PROJECT_PLAN.md     method, experiments, roadmap
acsid/README.md     architecture + cloud-A10 run guide
```

## Upstream provenance

`MiniOneRec/` is vendored **flat** (its own `.git` removed on 2026-08-19) from:

- origin: <https://github.com/AkaliKong/MiniOneRec.git>
- commit: `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`
- branch: `main` (2026-05-14)

Local edits on top of that commit live in `MiniOneRec/rq/` (`datasets.py`,
`rqvae.py`, `trainer.py`, `generate_indices.py`) and `MiniOneRec/requirements.txt`.
Run `git log -- MiniOneRec/rq/` to see them.

## Status

Phase 2 (SID construction) code complete; SFT/GRPO not yet adapted. The
pipeline is meant to execute on a cloud A10; see `acsid/README.md`.
