"""ACSID: Adaptive Collaborative Semantic ID construction.

Injects collaborative-filtering (Item2Vec) signals into the RQ-VAE input
stage via a learnable projection P and per-item adaptive weight alpha_i,
rather than at the RL reward stage. See PROJECT_PLAN.md for the full design.
"""

__all__ = ["item2vec", "adaptive_fusion", "generate_sid", "regenerate_csv_sid", "analyze_collision"]
