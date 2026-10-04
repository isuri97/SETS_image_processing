"""
Word-profile features compared with Dynamic Time Warping (Rath & Manmatha).

Each normalised word image (handwriting_norm.normalise_word) becomes a
sequence of per-column feature vectors. DTW aligns two sequences by
stretching/compressing them, so a letter written wider or a word shifted
sideways still aligns - the weakness of HOG's fixed grid.

Features per column (all scaled to ~[0, 1]):
    upper   : first ink row / height      (upper word contour)
    lower   : last ink row / height       (lower word contour)
    ink     : ink pixels / height         (vertical projection)
    trans   : ink/background transitions  (stroke count through the column)
    core    : ink fraction inside the x-height zone

All sequences are resampled to a fixed length so that every pair can be
compared in one vectorised DTW over the whole A x B score matrix.
"""

from __future__ import annotations

import numpy as np

from handwriting_norm import NormParams


def column_features(word: np.ndarray, p: NormParams = NormParams(),
                    length: int = 40) -> np.ndarray:
    """Normalised word image (ink = 1) -> (length, 5) feature sequence."""
    ink = word > 0.5
    h, w = ink.shape
    any_ink = ink.any(axis=0)
    rows = np.arange(h)[:, None]
    upper = np.where(any_ink, np.where(ink, rows, h).min(axis=0), np.nan) / h
    lower = np.where(any_ink, np.where(ink, rows, -1).max(axis=0), np.nan) / h
    # Columns without ink: interpolate the contours from their neighbours.
    for prof in (upper, lower):
        bad = np.isnan(prof)
        if bad.all():
            prof[:] = 0.5
        elif bad.any():
            prof[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), prof[~bad])
    count = ink.sum(axis=0) / h
    trans = np.abs(np.diff(ink.astype(np.int8), axis=0)).sum(axis=0) / 8.0
    c0, c1 = p.zones[0], p.zones[0] + p.zones[1]
    core = ink[c0:c1].sum(axis=0) / max(1, c1 - c0)
    feats = np.stack([upper, lower, count, trans, core], axis=1)

    # Resample to a fixed length (linear interpolation along x).
    x_old = np.linspace(0, 1, w)
    x_new = np.linspace(0, 1, length)
    return np.stack([np.interp(x_new, x_old, feats[:, k])
                     for k in range(feats.shape[1])], axis=1).astype(np.float32)


def dtw_matrix(X: np.ndarray, Y: np.ndarray, band: float = 0.2,
               chunk: int = 64) -> np.ndarray:
    """DTW distance between every sequence in X (n, L, F) and Y (m, L, F).

    Vectorised over all pairs: the DTW recursion runs over the L x L grid
    once, each step operating on an (n_chunk, m) array. A Sakoe-Chiba band
    of `band` x L limits how far the alignment may drift. Distances are
    divided by 2L so they are comparable across settings."""
    n, L, _ = X.shape
    r = max(1, int(band * L))
    out = np.empty((n, Y.shape[0]), np.float32)
    for s in range(0, n, chunk):
        Xc = X[s:s + chunk]
        prev = np.full((L + 1, Xc.shape[0], Y.shape[0]), np.inf, np.float32)
        prev[0] = 0.0
        for i in range(1, L + 1):
            cur = np.full_like(prev, np.inf)
            xi = Xc[:, i - 1, :]                                  # (nc, F)
            lo, hi = max(1, i - r), min(L, i + r)
            for j in range(lo, hi + 1):
                yj = Y[:, j - 1, :]                               # (m, F)
                c = np.sqrt(((xi[:, None, :] - yj[None, :, :]) ** 2).sum(-1))
                cur[j] = c + np.minimum(np.minimum(prev[j], cur[j - 1]), prev[j - 1])
            prev = cur
        out[s:s + chunk] = prev[L] / (2 * L)
    return out