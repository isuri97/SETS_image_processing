"""
Word spotting on segmented tokens: size shortlist -> HOG -> cosine similarity.

Builds on segment_manuscript.py (same folder). For every token:
1. Crop     : cut the token from the grayscale page and tighten it to its own
              ink (drops strokes intruding from neighbouring lines).
2. Normalise: invert (ink bright), stretch contrast, resize to a fixed canvas
              so every crop yields a feature vector of the same length.
3. HOG      : histogram of oriented gradients -> one vector per token.
4. Compare  : cosine similarity between mean-centred HOG vectors. Pairs are
              only scored if their box sizes pass a loose log-size shortlist
              (default 0.85), so size is a filter and appearance decides.
5. Group    : pairs above the visual threshold are linked; connected
              components become candidate "same word" groups.

Outputs (in --out_dir)
----------------------
- matches.csv       : every shortlisted pair with size and HOG similarity
- top_pairs.jpg     : the highest-scoring pairs side by side, for inspection
- spotting.jpg      : page overlay, one colour per visual group
- query_<id>.jpg    : ranked matches for a chosen token (--query c1_l12_t4)

Usage
-----
    python word_spotting.py 000126.jpg --out_dir spot
    python word_spotting.py 000126.jpg --query c2_l10_t4 --top_k 8
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

from segment_manuscript import (Box, Params, binarise, group_palette,
                                log_euclidean_similarity, segment)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SpotParams:
    size_shortlist: float = 0.85      # min log-size similarity to compare a pair
    visual_threshold: float = 0.60    # min HOG cosine to link a pair
    canvas: tuple = (48, 144)         # (height, width) of the normalised crop
    orientations: int = 9
    pixels_per_cell: tuple = (8, 8)
    cells_per_block: tuple = (2, 2)
    pad: int = 4                      # margin kept around the tightened ink


@dataclass
class Token:
    tid: str                          # e.g. "c1_l12_t4"
    box: Box
    crop: np.ndarray                  # normalised grayscale crop (canvas size)
    feat: np.ndarray | None = None


# --------------------------------------------------------------------------- #
# 1-2. Cropping and normalisation
# --------------------------------------------------------------------------- #
def tighten_to_ink(gray_crop: np.ndarray) -> np.ndarray:
    """Keep only ink components that overlap the vertical centre band of the
    crop; components living purely at the top/bottom edge are usually
    descenders/ascenders from adjacent lines. Returns the tightened crop."""
    ink = binarise(gray_crop)
    h = ink.shape[0]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    band0, band1 = int(0.3 * h), int(0.7 * h)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        top = stats[i, cv2.CC_STAT_TOP]
        bottom = top + stats[i, cv2.CC_STAT_HEIGHT]
        keep[i] = bottom > band0 and top < band1
    mask = keep[lab]
    if not mask.any():
        return gray_crop
    ys, xs = np.nonzero(mask)
    return gray_crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalise_crop(gray_crop: np.ndarray, p: SpotParams) -> np.ndarray:
    """Invert, contrast-stretch and pad to the canvas aspect, then resize.
    Aspect-preserving padding avoids distorting stroke directions (HOG is
    sensitive to them)."""
    img = 255 - gray_crop.astype(np.float32)
    lo, hi = np.percentile(img, (5, 99))
    img = np.clip((img - lo) / max(hi - lo, 1), 0, 1)

    ch, cw = p.canvas
    h, w = img.shape
    target_w = max(w, int(round(h * cw / ch)))        # pad width, or ...
    target_h = max(h, int(round(w * ch / cw)))        # ... pad height
    canvas = np.zeros((target_h, target_w), np.float32)
    y0, x0 = (target_h - h) // 2, (target_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    canvas = np.pad(canvas, p.pad)
    return cv2.resize(canvas, (cw, ch), interpolation=cv2.INTER_AREA)


def extract_tokens(image_path: str, sp: SpotParams) -> tuple[np.ndarray, list[Token]]:
    img, columns = segment(image_path, Params())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tokens = []
    for c, col in enumerate(columns, 1):
        for l, line in enumerate(col.lines, 1):
            for t, b in enumerate(line.tokens, 1):
                tight = tighten_to_ink(gray[b.y0:b.y1, b.x0:b.x1])
                tokens.append(Token(f"c{c}_l{l}_t{t}", b, normalise_crop(tight, sp)))
    return img, tokens


# --------------------------------------------------------------------------- #
# 3. Features
# --------------------------------------------------------------------------- #
def compute_hog(tokens: list[Token], sp: SpotParams) -> np.ndarray:
    """HOG per token, then mean-centred and L2-normalised.

    Centring matters: raw HOG values are non-negative, so raw cosine scores
    all sit near 1 and barely discriminate. Subtracting the page mean keeps
    only what makes each token different from the average token."""
    feats = np.stack([hog(t.crop, orientations=sp.orientations,
                          pixels_per_cell=sp.pixels_per_cell,
                          cells_per_block=sp.cells_per_block,
                          block_norm="L2-Hys") for t in tokens])
    feats -= feats.mean(axis=0, keepdims=True)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    for t, f in zip(tokens, feats):
        t.feat = f
    return feats


# --------------------------------------------------------------------------- #
# 4. Similarity
# --------------------------------------------------------------------------- #
def score_pairs(tokens: list[Token], feats: np.ndarray,
                sp: SpotParams) -> list[tuple[int, int, float, float]]:
    """(i, j, size_sim, hog_cosine) for every pair passing the size shortlist,
    sorted by HOG cosine (best first)."""
    cos = feats @ feats.T
    pairs = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            s = log_euclidean_similarity(tokens[i].box, tokens[j].box)
            if s >= sp.size_shortlist:
                pairs.append((i, j, s, float(cos[i, j])))
    return sorted(pairs, key=lambda x: -x[3])


def visual_groups(n: int, pairs, threshold: float) -> list[int]:
    """Union-find over pairs above `threshold`; returns a group id per token
    (-1 for tokens with no visual match)."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _, c in pairs:
        if c >= threshold:
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    sizes = np.bincount(roots, minlength=n)
    ids, out = {}, []
    for r in roots:
        if sizes[r] < 2:
            out.append(-1)
        else:
            out.append(ids.setdefault(r, len(ids)))
    return out


# --------------------------------------------------------------------------- #
# 5. Visualisation
# --------------------------------------------------------------------------- #
def _tile(crop: np.ndarray, label: str, scale: int = 2) -> np.ndarray:
    tile = cv2.cvtColor((255 - crop * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    tile = cv2.copyMakeBorder(tile, 22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return tile


def draw_top_pairs(tokens, pairs, k: int) -> np.ndarray:
    rows = []
    for i, j, s, c in pairs[:k]:
        a = _tile(tokens[i].crop, tokens[i].tid)
        b = _tile(tokens[j].crop, tokens[j].tid)
        info = np.full((a.shape[0], 150, 3), 255, np.uint8)
        cv2.putText(info, f"hog {c:.2f}", (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1)
        cv2.putText(info, f"size {s:.2f}", (8, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
        rows.append(cv2.hconcat([a, b, info]))
    return cv2.vconcat(rows)


def draw_query(tokens, feats, q: int, k: int, sp: SpotParams) -> np.ndarray:
    """Query token followed by its k best matches among size-shortlisted tokens."""
    cand = [j for j in range(len(tokens)) if j != q and
            log_euclidean_similarity(tokens[q].box, tokens[j].box) >= sp.size_shortlist]
    ranked = sorted(cand, key=lambda j: -float(feats[q] @ feats[j]))[:k]
    tiles = [_tile(tokens[q].crop, f"QUERY {tokens[q].tid}")]
    tiles += [_tile(tokens[j].crop, f"{tokens[j].tid} {float(feats[q] @ feats[j]):.2f}")
              for j in ranked]
    return cv2.vconcat(tiles)


def draw_overlay(img, tokens, groups) -> np.ndarray:
    out = img.copy()
    n = max(groups) + 1 if groups else 0
    colours = group_palette(n)
    th = max(3, img.shape[1] // 800)
    for t, g in zip(tokens, groups):
        if g >= 0:
            b = t.box
            cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), colours[g], th)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out_dir", default="spot_out")
    ap.add_argument("--size_shortlist", type=float, default=SpotParams.size_shortlist)
    ap.add_argument("--visual_threshold", type=float, default=SpotParams.visual_threshold)
    ap.add_argument("--top_k", type=int, default=15)
    ap.add_argument("--query", default=None, help="token id, e.g. c2_l10_t4")
    a = ap.parse_args()

    sp = SpotParams(size_shortlist=a.size_shortlist, visual_threshold=a.visual_threshold)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    img, tokens = extract_tokens(a.image, sp)
    feats = compute_hog(tokens, sp)
    pairs = score_pairs(tokens, feats, sp)
    groups = visual_groups(len(tokens), pairs, sp.visual_threshold)

    with open(out / "matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token_a", "token_b", "size_similarity", "hog_cosine"])
        for i, j, s, c in pairs:
            w.writerow([tokens[i].tid, tokens[j].tid, f"{s:.3f}", f"{c:.3f}"])
    cv2.imwrite(str(out / "top_pairs.jpg"), draw_top_pairs(tokens, pairs, a.top_k))
    cv2.imwrite(str(out / "spotting.jpg"), draw_overlay(img, tokens, groups),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    if a.query:
        ids = {t.tid: i for i, t in enumerate(tokens)}
        if a.query not in ids:
            raise SystemExit(f"unknown token id {a.query}; see matches.csv")
        cv2.imwrite(str(out / f"query_{a.query}.jpg"),
                    draw_query(tokens, feats, ids[a.query], a.top_k, sp))

    n_groups = max(groups) + 1 if groups else 0
    print(f"{len(tokens)} tokens, {len(pairs)} size-shortlisted pairs, "
          f"{sum(c >= sp.visual_threshold for *_, c in pairs)} visual links, "
          f"{n_groups} visual groups ({sum(g >= 0 for g in groups)} tokens)")


if __name__ == "__main__":
    main()