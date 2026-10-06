"""
Label-free improvements to HOG word matching.

1. Whitened HOG (LDA-style / closed-form exemplar classifier)
   Raw HOG dimensions are strongly correlated and many mostly encode paper
   texture or ink density shared by every crop. Whitening estimates the
   covariance of HOG vectors over all crops of both pages and rescales the
   principal directions to unit variance:
       x' = diag(1 / sqrt(var_k + eps)) . V_k^T (x - mu)
   so directions that vary for every crop are down-weighted and directions
   that separate words are emphasised. With the mean subtracted, the cosine
   of whitened vectors equals the score of the closed-form "exemplar LDA"
   classifier of Hariharan et al. (2012), used for word spotting by
   Almazan et al. (2012).

2. HOG-DTW
   HOG is computed on the normalised word image and read off as a sequence
   of block-columns (left to right). Sequences are reduced with PCA (fitted
   on both pages) and aligned with DTW, combining HOG's local gradient
   detail with DTW's tolerance to letters written wider or shifted - the
   main weakness of comparing HOG on a fixed grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog


@dataclass
class HogPlusParams:
    # whitening
    whiten_dims: int = 128            # principal directions kept
    whiten_eps: float = 0.1           # regulariser, fraction of mean kept variance
    # HOG-DTW
    seq_length: int = 32              # block-columns after resampling
    seq_dims: int = 24                # PCA dims per block-column
    min_width: int = 24               # pad narrow words to at least this (px)
    orientations: int = 9
    pixels_per_cell: tuple = (8, 8)
    cells_per_block: tuple = (2, 2)


# --------------------------------------------------------------------------- #
# Shared PCA helper
# --------------------------------------------------------------------------- #
def fit_pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, components (k, d), variances (k,)) of the rows of X."""
    mu = X.mean(axis=0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    k = min(k, vt.shape[0])
    var = (s[:k] ** 2) / max(1, X.shape[0] - 1)
    return mu, vt[:k], var


# --------------------------------------------------------------------------- #
# 1. Whitened HOG
# --------------------------------------------------------------------------- #
def raw_hog(crops: list[np.ndarray], hp: HogPlusParams) -> np.ndarray:
    return np.stack([hog(c, orientations=hp.orientations,
                         pixels_per_cell=hp.pixels_per_cell,
                         cells_per_block=hp.cells_per_block,
                         block_norm="L2-Hys") for c in crops]).astype(np.float64)


def whitened_hog(cands: list, hp: HogPlusParams = HogPlusParams()) -> np.ndarray:
    """Set c.feat to the whitened, L2-normalised HOG of each candidate.
    Statistics are pooled over all candidates passed (both pages)."""
    F = raw_hog([c.crop for c in cands], hp)
    mu, V, var = fit_pca(F, hp.whiten_dims)
    W = (F - mu) @ V.T / np.sqrt(var + hp.whiten_eps * var.mean())
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    for c, w in zip(cands, W):
        c.feat = w
    return W


# --------------------------------------------------------------------------- #
# 2. HOG block-column sequences for DTW
# --------------------------------------------------------------------------- #
def hog_column_sequence(word: np.ndarray, hp: HogPlusParams = HogPlusParams()) -> np.ndarray:
    """Normalised word image (fixed height, natural width) -> (seq_length, D)
    sequence of HOG block-column descriptors, resampled along x."""
    if word.shape[1] < hp.min_width:
        pad = hp.min_width - word.shape[1]
        word = np.pad(word, ((0, 0), (pad // 2, pad - pad // 2)))
    H = hog(word, orientations=hp.orientations, pixels_per_cell=hp.pixels_per_cell,
            cells_per_block=hp.cells_per_block, block_norm="L2-Hys",
            feature_vector=False)                         # (rows, cols, 2, 2, o)
    seq = H.transpose(1, 0, 2, 3, 4).reshape(H.shape[1], -1)  # one row per block-column
    x_old = np.linspace(0, 1, len(seq))
    x_new = np.linspace(0, 1, hp.seq_length)
    return np.stack([np.interp(x_new, x_old, seq[:, d]) for d in range(seq.shape[1])],
                    axis=1).astype(np.float32)


def project_sequences(cands: list, hp: HogPlusParams = HogPlusParams()) -> None:
    """PCA-reduce every candidate's HOG sequence (c.hseq, set in place).
    PCA is fitted on all block-columns of all candidates of both pages."""
    S = np.stack([c.hseq for c in cands])                 # (n, L, D)
    mu, V, var = fit_pca(S.reshape(-1, S.shape[2]).astype(np.float64), hp.seq_dims)
    P = ((S - mu) @ V.T) / np.sqrt(var.mean())            # keep relative variances
    for c, p in zip(cands, P.astype(np.float32)):
        c.hseq = p