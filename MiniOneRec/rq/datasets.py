import os

import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):
    """Original single-embing loader. Preserved for the stock RQ-VAE pipeline."""

    def __init__(self, data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)

        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0

        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}")

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class FusedEmbDataset(data.Dataset):
    """Dataset for ACSID collaborative-text fusion.

    Builds three aligned per-item arrays indexed by item_id == row index
    (same contract as the text .npy):

      - z_text : [N, D_text]  raw text embeddings (frozen)
      - z_cf   : [N, D_cf]    Item2Vec collaborative embeddings (train-only)
      - alpha  : [N]          per-item fusion weight in [0, alpha_max]
                              (alpha_i == 0 => cold-start => pure text)

    `__getitem__` returns a dict so the default PyTorch collate stacks each
    field into a batch along dim 0. The RQ-VAE trainer fuses the three
    batches at the start of each step via residual injection
    (``z = z_text + alpha * ||z_text|| * Normalize(P(z_cf))``); z_text is
    never L2-normalized. In text mode the FusionModule is not constructed
    and the trainer returns raw z_text unchanged, so all three modes share
    one code path and one batch shape.

    cf_path / alpha_path are optional: when missing or the file does not
    exist, the corresponding array is filled with zeros (alpha defaults to
    0, i.e. pure text), letting a caller run the data pipeline in text mode
    without producing intermediate collaborative artifacts.
    """

    def __init__(self, text_path, cf_path=None, alpha_path=None):
        self.text_path = text_path
        self.cf_path = cf_path
        self.alpha_path = alpha_path

        # --- text ---
        text = np.load(text_path)
        nan_mask = np.isnan(text)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in text embeddings")
            text[nan_mask] = 0.0
        inf_mask = np.isinf(text)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} inf values in text embeddings")
            text[inf_mask] = 0.0
        self.text = text.astype(np.float32)
        self.dim = self.text.shape[-1]  # kept for EmbDataset API parity
        n = self.text.shape[0]
        print(f"[FusedEmbDataset] text shape: {self.text.shape}")

        # --- cf ---
        if cf_path and os.path.exists(cf_path):
            cf = np.load(cf_path)
            bad = np.isnan(cf) | np.isinf(cf)
            if bad.any():
                print(f"Warning: Found {int(bad.sum())} NaN/Inf in cf, zeroed")
                cf[bad] = 0.0
            cf = cf.astype(np.float32).reshape(n, -1)
            if cf.shape[0] != n:
                raise ValueError(f"cf row count {cf.shape[0]} != text row count {n}")
        else:
            # 1-d dummy descriptor so default collate still yields [B, 1].
            cf = np.zeros((n, 1), dtype=np.float32)
        self.cf = cf
        self.cf_dim = self.cf.shape[-1]
        print(f"[FusedEmbDataset] cf shape: {self.cf.shape}")

        # --- alpha ---
        if alpha_path and os.path.exists(alpha_path):
            alpha = np.load(alpha_path).astype(np.float32).reshape(-1)
            if alpha.shape[0] != n:
                raise ValueError(f"alpha length {alpha.shape[0]} != text row count {n}")
        else:
            alpha = np.zeros(n, dtype=np.float32)
        self.alpha = alpha
        print(f"[FusedEmbDataset] alpha shape: {self.alpha.shape}, stats: "
              f"min={self.alpha.min():.4f} max={self.alpha.max():.4f} mean={self.alpha.mean():.4f}")

    @property
    def dim_text(self):
        return self.dim

    @property
    def dim_cf(self):
        return self.cf_dim

    def __getitem__(self, index):
        return {
            "z_text": torch.from_numpy(self.text[index]).float(),
            "z_cf": torch.from_numpy(self.cf[index]).float(),
            "alpha": torch.tensor([float(self.alpha[index])], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.text)
