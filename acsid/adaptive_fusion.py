"""Adaptive collaborative-text fusion for ACSID.

Implements the per-item adaptive weight from PROJECT_PLAN.md §2.3:

    alpha_i = alpha_max * min(1, log(1 + n_i) / log(1 + n_ref))
    z_hat_t = Normalize(z_text)
    z_hat_c = Normalize(P(z_cf))
    z_i     = Normalize[(1 - alpha_i) * z_hat_t + alpha_i * z_hat_c]

P is Linear(cf_dim, text_dim) trained jointly with RQ-VAE; its output is
L2-normalized so the scale of the CF embedding does not leak into alpha.
Item-id == row index (see PROJECT_PLAN.md §12.2), so alpha/cf arrays are
indexed by the same row used for the text .npy.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-item adaptive weight
# ---------------------------------------------------------------------------

def compute_item_freq(train_csv: str) -> np.ndarray:
    """Return per-item interaction count over the train split only.

    Reads only the train CSV so valid/test interactions never leak into
    the collaborative signal (see PROJECT_PLAN.md §12.1). The returned
    array is indexed by item_id == row index: count[i] = number of times
    item i appears in any train user's history or target.

    The CSV stores lists as pandas repr ("['117','130']" style); we eval
    them to recover the Python lists, matching data.py's SidSFTDataset.
    """
    import pandas as pd

    df = pd.read_csv(train_csv)

    # upper bound on item_id gives the size we need
    max_id = -1
    counts: dict[int, int] = {}

    def _bump(item_ids: Iterable[int]) -> None:
        nonlocal max_id
        for x in item_ids:
            x = int(x)
            counts[x] = counts.get(x, 0) + 1
            if x > max_id:
                max_id = x

    for hist, tgt in zip(df["history_item_id"], df["item_id"]):
        try:
            history_list = eval(hist) if isinstance(hist, str) else hist
        except Exception:
            history_list = []
        if history_list is None:
            history_list = []
        _bump(history_list)
        _bump([int(tgt)])

    n = max_id + 1 if max_id >= 0 else 0
    arr = np.zeros(max(n, 0), dtype=np.int64)
    for k, v in counts.items():
        arr[k] = v
    return arr


def compute_alpha(
    freq: np.ndarray,
    alpha_max: float = 0.3,
    n_ref: Optional[float] = None,
    fixed: bool = False,
) -> np.ndarray:
    """Per-item weight in [0, alpha_max].

    fixed=True : every item uses alpha_max (the "Fixed-CF" baseline; cold-start
                 items are NOT zeroed — that's the whole point of "fixed",
                 contrast with adaptive).
    fixed=False: adaptive formula
        alpha_i = alpha_max * min(1, log(1+n_i) / log(1+n_ref))
    n_ref defaults to the median frequency over items that have >=1 train
    interaction. Items never seen in train map to 0 (cold-start fallback).
    """
    freq = np.asarray(freq, dtype=np.float64)
    if freq.size == 0:
        return np.zeros(0, dtype=np.float32)
    if fixed:
        # everyone gets the same weight, including cold-start
        return np.full(freq.shape, float(alpha_max), dtype=np.float32)

    if n_ref is None:
        nonzero = freq[freq > 0]
        n_ref = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
    if n_ref <= 0:
        n_ref = 1.0  # guard; min(1,..) keeps ratio bounded regardless

    denom = math.log(1.0 + n_ref)
    if denom <= 0:
        denom = 1.0

    ratio = np.log(1.0 + freq) / denom
    coeff = np.minimum(1.0, ratio)
    alpha = alpha_max * coeff
    # cold-start: items with zero interactions get exactly 0
    alpha[freq <= 0] = 0.0
    # numerical safety clamp
    alpha = np.clip(alpha, 0.0, alpha_max)
    return alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion module: learnable projection P + L2-normalized weighted sum
# ---------------------------------------------------------------------------

class FusionModule(nn.Module):
    """Holds P and the adaptive-fusion arithmetic.

    Parameters:
        cf_dim   : input dim of the collaborative embedding (Item2Vec), e.g. 256
        text_dim : input dim of the text embedding (RQ-VAE in_dim), read
                   from the .npy at runtime; never hard-coded.
        bias     : whether P has a bias term. Plan says "Linear(256->2560)";
                   we keep bias=False by default since the L2-normalize after
                   P makes a bias term largely redundant and the original
                   spec omits it.

    forward(z_text, z_cf, alpha):
        z_text : [B, text_dim]  (raw, will be L2-normalized here)
        z_cf   : [B, cf_dim]
        alpha  : [B] or [B, 1]  (per-item weight in [0, alpha_max])
        returns z_i : [B, text_dim]  (L2-normalized fused vector)
    """

    def __init__(self, cf_dim: int, text_dim: int, bias: bool = False):
        super().__init__()
        self.cf_dim = cf_dim
        self.text_dim = text_dim
        self.P = nn.Linear(cf_dim, text_dim, bias=bias)

    def forward(self, z_text: torch.Tensor, z_cf: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if z_text.dim() != 2 or z_cf.dim() != 2:
            raise ValueError(f"expected 2D tensors, got z_text={tuple(z_text.shape)}, z_cf={tuple(z_cf.shape)}")
        if z_text.size(0) != z_cf.size(0) or z_text.size(0) != alpha.size(0):
            raise ValueError("batch mismatch between z_text, z_cf, alpha")
        if z_cf.size(1) != self.cf_dim:
            raise ValueError(f"z_cf has dim {z_cf.size(1)} but P expects {self.cf_dim}")

        z_hat_t = F.normalize(z_text, p=2, dim=-1)
        z_hat_c = F.normalize(self.P(z_cf), p=2, dim=-1)

        a = alpha.view(-1, 1)
        fused = (1.0 - a) * z_hat_t + a * z_hat_c
        return F.normalize(fused, p=2, dim=-1)


def fuse_full(
    fusion: FusionModule,
    z_text: torch.Tensor,
    z_cf: torch.Tensor,
    alpha: torch.Tensor,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Fuse all items at once on GPU in chunks; returns [N, text_dim] on CPU.

    Used by generate_indices.py to reconstruct the full fused embedding
    matrix before the RQ-VAE get_indices pass. No gradients.
    """
    fusion.eval()
    out = torch.empty((z_text.size(0), fusion.text_dim), dtype=z_text.dtype)
    with torch.no_grad():
        for s in range(0, z_text.size(0), batch_size):
            e = min(s + batch_size, z_text.size(0))
            t = z_text[s:e].to(device)
            c = z_cf[s:e].to(device)
            al = alpha[s:e].to(device)
            out[s:e] = fusion(t, c, al).cpu()
    return out
