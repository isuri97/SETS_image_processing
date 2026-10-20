"""
Cross-document word spotting over a whole folder of pages: which words recur
within each page, and which words on one page also occur on another?

Reuses segment_manuscript.py, word_spotting.py, handwriting_norm.py,
hog_plus.py and profile_dtw.py.

Pipeline
--------
1. Segment every page into tokens (segment_manuscript.segment).
2. Candidate expansion  [fix for blocked size checks]
   a) Over-wide tokens (several words merged) are re-split at their widest
      internal gaps until every piece is word-sized.
   b) Edge trimming: a short piece at the start or end of a token (e.g. the
      "S." of "S. Cassan", a trailing flourish) is also removed, giving extra
      candidates "Cassan" and "S. Cassan". Every candidate stays word-length;
      fragments shorter than `min_match_lines` are never reported as matches
      (two-letter pieces resemble everything) but stay in the background
      statistics used for the z-scores.
3. Handwriting normalisation (handwriting_norm): slant removal, x-height
   zone alignment and fixed stroke width, so pen and scribe differences
   between pages are reduced before any comparison.
4. Two complementary comparisons of every candidate pair:
   - HOG cosine on the normalised crop, whitened with statistics pooled
     over ALL pages in the run (hog_plus.whitened_hog);
   - DTW distance between column-profile sequences (profile_dtw), which
     tolerates letters written wider or shifted sideways;
   - optionally, HOG block-column sequences aligned by DTW
     (hog_plus, `hog_dtw_weight`; off by default).
   Each is standardised over all pairs (global z) and the two are averaged
   into one score, weighted by `dtw_weight`.
5. Token-level scores: for each (A token, B token) the best combined score
   over their size-compatible candidate pairs (sizes normalised by line
   pitch).
6. Query-adaptive acceptance  [fix for the single global threshold]
   A pair (a, b) is a match if
     - its z-score against all of a's scores AND against all of b's scores
       is >= `z_threshold` (background over all token pairs, before the
       size filter), and
     - its score is within `margin` (global SD units) of the best score of
       query a AND of query b, and
     - a is in b's top-k and b is in a's top-k (reciprocal top-k), and
     - its score >= `score_floor` (global SD units).
7. Phrase context  [formulaic records]
   Support for (a, b) is the best z over neighbour pairs on the same side,
   up to `context_reach` tokens away in reading order. The ranking score is
       z_ctx = z + `context_weight` x clip(support, 0, `context_cap`)
   and acceptance uses z_ctx.

Multi-page run
--------------
For N pages the pipeline runs
  - N within-page comparisons   (match(P, P), if `within`)
  - N(N-1)/2 between-page ones  (match(P, Q) for every pair, if `cross`)
and links all accepted matches into word clusters across the whole folder.

Outputs (in --out_dir)
----------------------
- within_<page>.csv/.jpg        : matches inside each page
- cross_<pageA>__<pageB>.csv/.jpg : matches between each pair of pages
- summary.csv                   : number of matches per comparison
- clusters.csv/.jpg             : all matches linked into groups of the same word
- clusters_<page>.jpg           : each page with its cluster occurrences marked
                                  (colours and numbers consistent across pages)
- clusters_overlay.jpg          : all pages side by side (only when there are
                                  at most `max_overlay_pages` pages)

Usage
-----
    python cross_spotting.py data/ --out_dir out                # every image in data/
    python cross_spotting.py a.jpg b.jpg c.jpg --out_dir out    # explicit list
    python cross_spotting.py data/ --no_within                  # between pages only
    python cross_spotting.py 000126.jpg --out_dir out           # one page only
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from core.comparison_of_two_documents.experiment_1.segment_manuscript import Box, Params, _runs, binarise, group_palette, segment
from core.comparison_of_two_documents.experiment_1.handwriting_norm import normalise_word, to_canvas
from core.comparison_of_two_documents.experiment_2.hog_plus import (HogPlusParams, hog_column_sequence, project_sequences,
                                                                    whitened_hog)
from core.comparison_of_two_documents.experiment_1.profile_dtw import column_features, dtw_matrix
from core.comparison_of_two_documents.experiment_1.word_spotting import SpotParams, compute_hog, tighten_to_ink


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# --------------------------------------------------------------------------- #
# Configuration and data
# --------------------------------------------------------------------------- #
@dataclass
class CrossParams:
    # candidate expansion (lengths in line pitches)
    max_width_lines: float = 4.0      # wider tokens are re-split
    min_width_lines: float = 0.6      # narrower candidates are dropped
    min_match_lines: float = 1.2      # narrower candidates are background only:
                                      # they shape the z-score statistics but
                                      # can never be reported as a match
    edge_piece_lines: float = 1.0     # max width of a trimmable edge piece
    min_split_gap: float = 0.06       # min internal gap usable as a cut
    # matching
    size_shortlist: float = 0.80      # log-size similarity on normalised boxes
    z_threshold: float = 3.25         # min (context) z-score vs background
    margin: float = 0.75              # keep matches within margin of query's best
    top_k: int = 3                    # reciprocal top-k (between pages)
    top_k_within: int = 5             # reciprocal top-k within one page (a
                                      # word may recur many times on a page)
    score_floor: float = 1.5          # min combined score (global SD units)
    # features (weights are relative; components with weight 0 are skipped)
    whiten: bool = True               # whitened HOG instead of mean-centred HOG
    hog_weight: float = 0.5           # HOG cosine
    dtw_weight: float = 0.5           # column-profile DTW
    hog_dtw_weight: float = 0.0       # HOG block-column sequences + DTW
    # phrase context
    context_weight: float = 0.35      # z bonus per unit of neighbour support
    context_cap: float = 3.0          # max neighbour support counted
    context_reach: int = 2            # neighbours up to this many tokens away


@dataclass
class Candidate:
    tid: str                          # e.g. "c1_l9_t6[0:3]"
    parent: str                       # e.g. "c1_l9_t6"
    box: Box                          # page coordinates of the span
    crop: np.ndarray                  # normalised crop fed to HOG
    seq: np.ndarray                   # column-profile sequence fed to DTW
    raw: np.ndarray                   # original tight crop, for display only
    feat: np.ndarray | None = None
    hseq: np.ndarray | None = None    # HOG block-column sequence (HOG-DTW)
    is_full: bool = False             # True if the span is the whole token


@dataclass
class Page:
    name: str
    img: np.ndarray
    unit: float                       # line pitch in px
    cands: list[Candidate] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Input collection
# --------------------------------------------------------------------------- #
def collect_images(inputs: str | Path | list[str | Path]) -> list[Path]:
    """Expand files and folders into a sorted, de-duplicated list of image
    paths. Folders are scanned non-recursively for IMAGE_EXTS."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    found: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found += sorted(f for f in p.iterdir()
                            if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(f"image or folder not found: {p}")
    images = list(dict.fromkeys(f.resolve() for f in found))
    stems = [f.stem for f in images]
    dupes = {s for s in stems if stems.count(s) > 1}
    if dupes:                         # page names key the outputs and clusters
        raise ValueError(f"several images share the same file name stem: {sorted(dupes)}")
    if not images:
        raise FileNotFoundError(f"no images ({', '.join(sorted(IMAGE_EXTS))}) found in {inputs}")
    return images


# --------------------------------------------------------------------------- #
# 2. Candidate expansion
# --------------------------------------------------------------------------- #
def internal_gaps(gray_crop: np.ndarray, unit: float, cp: CrossParams) -> list[tuple[int, int]]:
    """(width, x_mid) of internal white gaps in the core band, crop-local."""
    ink = binarise(gray_crop)
    h = ink.shape[0]
    runs = _runs(ink[int(0.2 * h):int(0.8 * h)].sum(axis=0) > 0)
    return [(b0 - a1, (a1 + b0) // 2)
            for (_, a1), (b0, _) in zip(runs[:-1], runs[1:])
            if b0 - a1 >= cp.min_split_gap * unit]


def split_wide(x0: int, x1: int, gaps: list[tuple[int, int]], unit: float,
               cp: CrossParams) -> list[tuple[int, int]]:
    """Recursively cut [x0, x1) at its widest gap until pieces are word-sized."""
    if x1 - x0 <= cp.max_width_lines * unit:
        return [(x0, x1)]
    inside = [g for g in gaps if x0 < g[1] < x1]
    if not inside:
        return [(x0, x1)]
    _, cut = max(inside)
    return split_wide(x0, cut, gaps, unit, cp) + split_wide(cut, x1, gaps, unit, cp)


def edge_trims(x0: int, x1: int, gaps: list[tuple[int, int]], unit: float,
               cp: CrossParams) -> list[tuple[int, int]]:
    """The span itself plus versions with a short first/last piece removed."""
    cuts = sorted(g[1] for g in gaps if x0 < g[1] < x1)
    lim = cp.edge_piece_lines * unit
    starts = [x0] + ([cuts[0]] if cuts and cuts[0] - x0 <= lim else [])
    ends = [x1] + ([cuts[-1]] if cuts and x1 - cuts[-1] <= lim else [])
    return [(a, b) for a in starts for b in ends if b > a]


def expand_token(gray: np.ndarray, b: Box, tid: str, unit: float,
                 cp: CrossParams, sp: SpotParams) -> list[Candidate]:
    crop = gray[b.y0:b.y1, b.x0:b.x1]
    w = crop.shape[1]
    gaps = internal_gaps(crop, unit, cp)
    spans = []
    for p0, p1 in split_wide(0, w, gaps, unit, cp):
        spans += edge_trims(p0, p1, gaps, unit, cp)
    out = []
    for x0, x1 in dict.fromkeys(spans):                  # dedupe, keep order
        if not cp.min_width_lines * unit <= x1 - x0 <= cp.max_width_lines * unit:
            continue
        full = (x0, x1) == (0, w)
        tight = tighten_to_ink(crop[:, x0:x1])
        word = normalise_word(tight)
        out.append(Candidate(
            tid=tid if full else f"{tid}[{x0}:{x1}]", parent=tid,
            box=Box(b.x0 + x0, b.y0, b.x0 + x1, b.y1),
            crop=to_canvas(word), seq=column_features(word), raw=tight,
            hseq=hog_column_sequence(word), is_full=full))
    return out


def load_page(image_path: str, cp: CrossParams, sp: SpotParams) -> Page:
    img, columns = segment(image_path, Params())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tokens = [(f"c{c}_l{l}_t{t}", b)
              for c, col in enumerate(columns, 1)
              for l, line in enumerate(col.lines, 1)
              for t, b in enumerate(line.tokens, 1)]
    page = Page(Path(image_path).stem, img, 0.0)
    if not tokens:
        return page
    page.unit = float(np.median([b.height for _, b in tokens]))
    for tid, b in tokens:
        page.cands += expand_token(gray, b, tid, page.unit, cp, sp)
    return page


# --------------------------------------------------------------------------- #
# 4-6. Matching
# --------------------------------------------------------------------------- #
def size_mask(A: Page, B: Page, cp: CrossParams) -> np.ndarray:
    def logsize(p):
        return np.array([[np.log(c.box.width / p.unit), np.log(c.box.height / p.unit)]
                         for c in p.cands])
    la, lb = logsize(A), logsize(B)
    d = np.linalg.norm(la[:, None, :] - lb[None, :, :], axis=2)
    return np.exp(-d) >= cp.size_shortlist


def _token_max(cos: np.ndarray, ra: np.ndarray, rb: np.ndarray, n_a: int, n_b: int):
    """Collapse a candidate-level matrix to tokens: best entry per token pair,
    plus the candidate indices (i, j) that achieved it."""
    T = np.full((n_a, n_b), -1.0)
    arg = np.zeros((n_a, n_b, 2), int)
    cols_of = [np.flatnonzero(rb == q) for q in range(n_b)]
    for p in range(n_a):
        rows = np.flatnonzero(ra == p)
        sub = cos[rows]
        for q, cols in enumerate(cols_of):
            block = sub[:, cols]
            k = int(block.argmax())
            T[p, q] = block.flat[k]
            arg[p, q] = rows[k // len(cols)], cols[k % len(cols)]
    return T, arg


def _gz(M: np.ndarray) -> np.ndarray:
    """Global standardisation (z-score over all entries)."""
    return (M - M.mean()) / (M.std() + 1e-8)


def prepare_features(pages: list[Page], cp: CrossParams, sp: SpotParams,
                     hp: HogPlusParams = HogPlusParams()) -> None:
    """Fit feature statistics (HOG centring or whitening, PCA of HOG
    sequences) on the candidates of all pages together, so every pairwise
    comparison uses the same feature space."""
    cands = [c for P in pages for c in P.cands]
    if cp.whiten:
        whitened_hog(cands, hp)
    else:
        compute_hog(cands, sp)
    if cp.hog_dtw_weight > 0:
        project_sequences(cands, hp)


def pair_scores(A: Page, B: Page, cp: CrossParams) -> dict[str, np.ndarray]:
    """Candidate-level matrices for each enabled component and their
    combination: every component is globally standardised (DTW distances
    negated, so higher = more similar) and averaged with its weight."""
    fa = np.stack([c.feat for c in A.cands]); fb = np.stack([c.feat for c in B.cands])
    out = {"cos": fa @ fb.T}
    parts = [(cp.hog_weight, out["cos"])]
    out["dtw"] = np.full_like(out["cos"], np.nan)
    if cp.dtw_weight > 0:
        out["dtw"] = dtw_matrix(np.stack([c.seq for c in A.cands]),
                                np.stack([c.seq for c in B.cands]))
        parts.append((cp.dtw_weight, -out["dtw"]))
    if cp.hog_dtw_weight > 0:
        out["hog_dtw"] = dtw_matrix(np.stack([c.hseq for c in A.cands]),
                                    np.stack([c.hseq for c in B.cands]))
        parts.append((cp.hog_dtw_weight, -out["hog_dtw"]))
    total = sum(w for w, _ in parts if w > 0)
    out["score"] = sum(w * _gz(m) for w, m in parts if w > 0) / total
    return out


def token_scores(A: Page, B: Page, S: np.ndarray, cp: CrossParams,
                 same_page: bool = False):
    """Token-level scores from the candidate-level score matrix S.

    T   : best score per (A token, B token) over size-compatible,
          word-length candidates
    U   : the same over ALL candidates, without size or width filters; used
          only as the background distribution for z-scores
    arg : candidate indices behind each entry of T

    With same_page=True (A is B) a token's score with itself is removed:
    -inf in T (never a match) and NaN in U (left out of the background)."""
    a_ids = list(dict.fromkeys(c.parent for c in A.cands))
    b_ids = list(dict.fromkeys(c.parent for c in B.cands))
    ra = np.array([a_ids.index(c.parent) for c in A.cands])
    rb = np.array([b_ids.index(c.parent) for c in B.cands])
    U, _ = _token_max(S, ra, rb, len(a_ids), len(b_ids))
    wa = np.array([c.box.width >= cp.min_match_lines * A.unit for c in A.cands])
    wb = np.array([c.box.width >= cp.min_match_lines * B.unit for c in B.cands])
    allowed = size_mask(A, B, cp) & wa[:, None] & wb[None, :]
    T, arg = _token_max(np.where(allowed, S, -np.inf), ra, rb, len(a_ids), len(b_ids))
    if same_page:
        np.fill_diagonal(T, -np.inf)
        np.fill_diagonal(U, np.nan)
    return T, U, arg


def _reciprocal_z(M: np.ndarray, U: np.ndarray) -> np.ndarray:
    """min(row z, column z) of M against the background U (NaNs ignored)."""
    with np.errstate(invalid="ignore"):
        zr = (M - np.nanmean(U, 1, keepdims=True)) / (np.nanstd(U, 1, keepdims=True) + 1e-8)
        zc = (M - np.nanmean(U, 0, keepdims=True)) / (np.nanstd(U, 0, keepdims=True) + 1e-8)
    return np.minimum(zr, zc)


def _reading_neighbours(ids: list[str], d: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of the token d places before / after each token in reading
    order (-1 if none). Neighbours never cross a column boundary but do
    cross lines."""
    col = [t.split("_")[0] for t in ids]
    n = len(ids)
    left = np.array([k - d if k - d >= 0 and col[k - d] == col[k] else -1 for k in range(n)])
    right = np.array([k + d if k + d < n and col[k + d] == col[k] else -1 for k in range(n)])
    return left, right


def context_support(Zn: np.ndarray, a_ids: list[str], b_ids: list[str],
                    reach: int) -> np.ndarray:
    """support[p, q] = best Zn over neighbour pairs on the same side, with
    offsets d1, d2 in 1..reach (tolerates an extra token on one page)."""
    support = np.full(Zn.shape, -np.inf)
    for d1 in range(1, reach + 1):
        la, ra = _reading_neighbours(a_ids, d1)
        for d2 in range(1, reach + 1):
            lb, rb = _reading_neighbours(b_ids, d2)
            for na, nb in ((la, lb), (ra, rb)):
                vals = Zn[np.ix_(np.maximum(na, 0), np.maximum(nb, 0))]
                vals = np.where((na[:, None] >= 0) & (nb[None, :] >= 0), vals, -np.inf)
                support = np.maximum(support, vals)
    return support


def match(A: Page, B: Page, cp: CrossParams) -> list[dict]:
    """Accepted matches between page A and page B. If B is A, the page is
    compared with itself: self-matches are excluded, each pair is reported
    once (p < q), and the reciprocal limit is `top_k_within`."""
    same = B is A
    P = pair_scores(A, B, cp)
    T, U, arg = token_scores(A, B, P["score"], cp, same_page=same)
    z = _reciprocal_z(T, U)
    a_ids = list(dict.fromkeys(c.parent for c in A.cands))
    b_ids = list(dict.fromkeys(c.parent for c in B.cands))
    zn = np.nan_to_num(_reciprocal_z(U, U), nan=-np.inf)
    support = context_support(zn, a_ids, b_ids, cp.context_reach)
    zc = z + cp.context_weight * np.clip(support, 0, cp.context_cap)

    k = cp.top_k_within if same else cp.top_k
    top_a = np.argsort(-T, axis=1)[:, :k]                 # best B tokens per A
    top_b = np.argsort(-T, axis=0)[:k, :]                 # best A tokens per B
    best_a, best_b = T.max(axis=1), T.max(axis=0)

    pairs = []
    for p, q in zip(*np.nonzero((zc >= cp.z_threshold) & (T >= cp.score_floor))):
        if same and p >= q:                               # report each pair once
            continue
        s = T[p, q]
        if (q in top_a[p] and p in top_b[:, q]
                and s >= best_a[p] - cp.margin and s >= best_b[q] - cp.margin):
            i, j = arg[p, q]
            pairs.append(dict(i=int(i), j=int(j), z=float(zc[p, q]), z_base=float(z[p, q]),
                              support=float(max(support[p, q], 0.0)), score=float(s),
                              cos=float(P["cos"][i, j]), dtw=float(P["dtw"][i, j])))
    return sorted(pairs, key=lambda d: -d["z"])


# --------------------------------------------------------------------------- #
# Word clusters across all comparisons
# --------------------------------------------------------------------------- #
@dataclass
class Cluster:
    members: list[tuple[Page, int]]   # (page, candidate index), reading order
    best_z: float                     # strongest link inside the cluster


def cluster_matches(results: list[tuple[Page, Page, list[dict]]],
                    overlap: float = 0.5) -> list[Cluster]:
    """Link all accepted matches (within and between pages) into groups of
    the same word with union-find. Two candidates of the same token whose
    spans overlap by >= `overlap` (x-IoU) are treated as one occurrence."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}
    pages: dict[str, Page] = {}
    best: dict[tuple[str, int], float] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for P, Q, pairs in results:
        pages[P.name], pages[Q.name] = P, Q
        for m in pairs:
            a, b = (P.name, m["i"]), (Q.name, m["j"])
            union(a, b)
            best[a] = max(best.get(a, -np.inf), m["z"]); best[b] = max(best.get(b, -np.inf), m["z"])

    # merge overlapping occurrences of the same token (only nodes of the same
    # page and token can overlap, so group by that first)
    by_parent: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for n in list(parent):
        by_parent.setdefault((n[0], pages[n[0]].cands[n[1]].parent), []).append(n)
    for group in by_parent.values():
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                (pa, ia), (pb, ib) = group[x], group[y]
                ca, cb = pages[pa].cands[ia], pages[pb].cands[ib]
                inter = max(0, min(ca.box.x1, cb.box.x1) - max(ca.box.x0, cb.box.x0))
                union_w = max(ca.box.x1, cb.box.x1) - min(ca.box.x0, cb.box.x0)
                if union_w and inter / union_w >= overlap:
                    union(group[x], group[y])

    groups: dict = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    clusters = []
    for members in groups.values():
        # one entry per token occurrence (keep the widest candidate)
        by_token: dict = {}
        for pname, i in members:
            key = (pname, pages[pname].cands[i].parent)
            if key not in by_token or pages[pname].cands[i].box.width > \
                    pages[pname].cands[by_token[key]].box.width:
                by_token[key] = i
        occ = sorted(((pages[pn], i) for (pn, _), i in by_token.items()),
                     key=lambda m: (m[0].name, m[1]))
        if len(occ) > 1:
            clusters.append(Cluster(occ, max(best[n] for n in members)))
    return sorted(clusters, key=lambda c: (-len(c.members), -c.best_z))


# --------------------------------------------------------------------------- #
# Visualisation and output
# --------------------------------------------------------------------------- #
def _raw_tile(c: Candidate, label: str, h: int = 70, w: int = 260) -> np.ndarray:
    """Original crop scaled to fit h x w on white, with a label above."""
    img = c.raw
    s = min(h / img.shape[0], w / img.shape[1])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    tile = np.full((h, w), 255, np.uint8)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    tile[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    tile = cv2.copyMakeBorder(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR), 22, 4, 4, 4,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return tile


def cp_context_bonus(p: dict) -> float:
    """Context bonus actually applied to a pair (z_ctx - z)."""
    return p["z"] - p.get("z_base", p["z"])


def draw_pairs(A: Page, B: Page, pairs: list[dict]) -> np.ndarray:
    rows = []
    for p in pairs:
        a = _raw_tile(A.cands[p["i"]], f"{A.name}:{A.cands[p['i']].tid}")
        b = _raw_tile(B.cands[p["j"]], f"{B.name}:{B.cands[p['j']].tid}")
        info = np.full((a.shape[0], 150, 3), 255, np.uint8)
        cv2.putText(info, f"z   {p['z']:.1f}", (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1)
        cv2.putText(info, f"hog {p['cos']:.2f}", (8, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        if not np.isnan(p["dtw"]):
            cv2.putText(info, f"dtw {p['dtw']:.3f}", (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        if p.get("support", 0) > 0:
            cv2.putText(info, f"ctx +{cp_context_bonus(p):.1f}", (80, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 0), 1)
        rows.append(cv2.hconcat([a, b, info]))
    return cv2.vconcat(rows) if rows else np.full((50, 400, 3), 255, np.uint8)


def draw_side_by_side(A: Page, B: Page, pairs: list[dict], height: int = 2000) -> np.ndarray:
    """Pairs sharing the same A span get one colour and number on both pages."""
    sa, sb = height / A.img.shape[0], height / B.img.shape[0]
    pa = cv2.resize(A.img, None, fx=sa, fy=sa)
    pb = cv2.resize(B.img, None, fx=sb, fy=sb)
    keys = list(dict.fromkeys(p["i"] for p in pairs))
    colours = dict(zip(keys, group_palette(len(keys))))
    for p in pairs:
        col, k = colours[p["i"]], keys.index(p["i"]) + 1
        for page, cand, s in ((pa, A.cands[p["i"]], sa), (pb, B.cands[p["j"]], sb)):
            b = cand.box
            x0, y0, x1, y1 = (int(v * s) for v in (b.x0, b.y0, b.x1, b.y1))
            cv2.rectangle(page, (x0, y0), (x1, y1), col, 4)
            cv2.putText(page, str(k), (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return cv2.hconcat([pa, np.full((height, 30, 3), 255, np.uint8), pb])


def draw_clusters(clusters: list[Cluster], per_row: int = 6) -> np.ndarray:
    """One block per cluster: its occurrences (original crops) side by side."""
    blocks = []
    for n, cl in enumerate(clusters, 1):
        tiles = [_raw_tile(P.cands[i], f"{P.name}:{P.cands[i].parent}", w=200)
                 for P, i in cl.members]
        while len(tiles) % per_row:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [cv2.hconcat(tiles[r:r + per_row]) for r in range(0, len(tiles), per_row)]
        head = np.full((26, rows[0].shape[1], 3), 235, np.uint8)
        n_pages = len({P.name for P, _ in cl.members})
        cv2.putText(head, f"cluster {n}: {len(cl.members)} occurrences on {n_pages} page(s), "
                          f"best z {cl.best_z:.1f}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        blocks.append(cv2.vconcat([head] + rows))
    if not blocks:
        return np.full((50, 400, 3), 255, np.uint8)
    w = max(b.shape[1] for b in blocks)
    return cv2.vconcat([cv2.copyMakeBorder(b, 0, 8, 0, w - b.shape[1], cv2.BORDER_CONSTANT,
                                           value=(255, 255, 255)) for b in blocks])


def _mark_clusters(P: Page, clusters: list[Cluster], palette: list, height: int) -> np.ndarray:
    """One page, resized to `height`, with every cluster occurrence boxed in
    the cluster's colour and numbered with the cluster index."""
    s = height / P.img.shape[0]
    canvas = cv2.resize(P.img, None, fx=s, fy=s)
    for n, (cl, col) in enumerate(zip(clusters, palette), 1):
        for Q, i in cl.members:
            if Q is not P:
                continue
            b = P.cands[i].box
            x0, y0, x1, y1 = (int(v * s) for v in (b.x0, b.y0, b.x1, b.y1))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), col, 4)
            cv2.putText(canvas, str(n), (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return canvas


def draw_clusters_on_pages(pages: list[Page], clusters: list[Cluster],
                           height: int = 2000) -> np.ndarray:
    """All pages side by side; every occurrence of a cluster gets the
    cluster's colour and number."""
    palette = group_palette(len(clusters))
    gap = np.full((height, 30, 3), 255, np.uint8)
    parts = []
    for P in pages:
        parts += [_mark_clusters(P, clusters, palette, height), gap]
    return cv2.hconcat(parts[:-1])


def write_matches(path: Path, A: Page, B: Page, pairs: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{A.name}_candidate", f"{B.name}_candidate", "z_context",
                    "z_score", "neighbour_support", "combined_score", "hog_cosine",
                    "dtw_distance"])
        for p in pairs:
            w.writerow([A.cands[p["i"]].tid, B.cands[p["j"]].tid, f"{p['z']:.2f}",
                        f"{p['z_base']:.2f}", f"{p['support']:.2f}", f"{p['score']:.2f}",
                        f"{p['cos']:.3f}", f"{p['dtw']:.3f}"])


def write_clusters(path: Path, clusters: list[Cluster]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "n_pages", "best_z", "page", "token", "candidate",
                    "x0", "y0", "x1", "y1"])
        for n, cl in enumerate(clusters, 1):
            n_pages = len({P.name for P, _ in cl.members})
            for P, i in cl.members:
                c = P.cands[i]
                w.writerow([n, len(cl.members), n_pages, f"{cl.best_z:.2f}", P.name, c.parent,
                            c.tid, c.box.x0, c.box.y0, c.box.x1, c.box.y1])


def write_summary(path: Path, results: list[tuple[Page, Page, list[dict]]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["page_a", "page_b", "type", "n_matches", "n_context_only"])
        for P, Q, pairs in results:
            w.writerow([P.name, Q.name, "within" if P is Q else "cross", len(pairs),
                        sum(p["z_base"] < p["z"] and p["support"] > 0 for p in pairs)])


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_spotting(images: str | Path | list[str | Path],
                 out_dir: str | Path = "spotting_out",
                 cp: CrossParams | None = None,
                 within: bool = True, cross: bool = True,
                 max_overlay_pages: int = 4,
                 verbose: bool = True) -> dict:
    """Find similar words within each page and between every pair of pages.

    `images` may be a folder, a single image, or a list of images/folders:

        from cross_spotting import run_spotting
        res = run_spotting("data/", "out")                        # whole folder
        res = run_spotting(["000126.jpg", "mstest.jpg"], "out")   # explicit pages

    Writes to `out_dir` (see the module docstring for the file list).
    Returns {"pages": [Page, ...], "pairs": [(P, Q, matches), ...],
             "clusters": [Cluster, ...]}.
    """
    cp = cp or CrossParams()
    sp = SpotParams()
    paths = collect_images(images)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1-3. load and segment every page once
    pages: list[Page] = []
    for n, path in enumerate(paths, 1):
        if verbose:
            print(f"[{n}/{len(paths)}] loading {path.name}")
        try:
            P = load_page(str(path), cp, sp)
        except Exception as e:                            # keep going on bad pages
            print(f"  skipped {path.name}: {type(e).__name__}: {e}")
            continue
        if not P.cands:
            print(f"  skipped {path.name}: no word candidates found")
            continue
        pages.append(P)
        if verbose:
            print(f"  {len(P.cands)} candidates, line pitch {P.unit:.0f}px")
    if not pages:
        raise RuntimeError("no usable pages")
    if len(pages) == 1:
        within, cross = True, False

    prepare_features(pages, cp, sp)                       # pooled over all pages

    # 4-7. within-page and between-page matching
    results = []
    if within:
        for P in pages:
            pairs = match(P, P, cp)
            results.append((P, P, pairs))
            write_matches(out / f"within_{P.name}.csv", P, P, pairs)
            cv2.imwrite(str(out / f"within_{P.name}.jpg"), draw_pairs(P, P, pairs))
            if verbose:
                print(f"{P.name}: {len(pairs)} matches within the page")
    if cross:
        todo = list(combinations(pages, 2))
        for n, (A, B) in enumerate(todo, 1):
            pairs = match(A, B, cp)
            results.append((A, B, pairs))
            stem = f"cross_{A.name}__{B.name}"
            write_matches(out / f"{stem}.csv", A, B, pairs)
            cv2.imwrite(str(out / f"{stem}.jpg"), draw_pairs(A, B, pairs))
            if verbose:
                print(f"[{n}/{len(todo)}] {A.name} <-> {B.name}: {len(pairs)} matches")

    # clusters over the whole folder
    clusters = cluster_matches(results)
    write_summary(out / "summary.csv", results)
    write_clusters(out / "clusters.csv", clusters)
    cv2.imwrite(str(out / "clusters.jpg"), draw_clusters(clusters))
    palette = group_palette(len(clusters))
    for P in pages:
        cv2.imwrite(str(out / f"clusters_{P.name}.jpg"),
                    _mark_clusters(P, clusters, palette, 2000), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if len(pages) <= max_overlay_pages:
        cv2.imwrite(str(out / "clusters_overlay.jpg"), draw_clusters_on_pages(pages, clusters),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    if verbose:
        multi = sum(len({P.name for P, _ in c.members}) > 1 for c in clusters)
        print(f"{len(pages)} pages, {len(clusters)} word clusters "
              f"({sum(len(c.members) for c in clusters)} occurrences, "
              f"{multi} spanning more than one page); outputs written to {out.resolve()}")
    return {"pages": pages, "pairs": results, "clusters": clusters}


def run_cross_spotting(image_a: str | Path, image_b: str | Path,
                       out_dir: str | Path = "cross_out",
                       cp: CrossParams | None = None,
                       verbose: bool = True) -> list[dict]:
    """Two-page comparison only (kept for earlier code). Writes
    cross_matches.csv, cross_matches.jpg and cross_overlay.jpg."""
    cp = cp or CrossParams()
    sp = SpotParams()
    for img in (image_a, image_b):
        if not Path(img).is_file():
            raise FileNotFoundError(f"image not found: {img}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    A, B = load_page(str(image_a), cp, sp), load_page(str(image_b), cp, sp)
    prepare_features([A, B], cp, sp)
    pairs = match(A, B, cp)

    write_matches(out / "cross_matches.csv", A, B, pairs)
    cv2.imwrite(str(out / "cross_matches.jpg"), draw_pairs(A, B, pairs))
    cv2.imwrite(str(out / "cross_overlay.jpg"), draw_side_by_side(A, B, pairs),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    if verbose:
        print(f"{A.name} <-> {B.name}: {len(pairs)} matches; "
              f"{sum(p['z_base'] < cp.z_threshold for p in pairs)} admitted only through "
              f"phrase context; outputs written to {out.resolve()}")
    return pairs


def main() -> None:
    """Command-line entry point (used when arguments are given)."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="image files and/or folders (folders are scanned for images)")
    ap.add_argument("--out_dir", default="spotting_out")
    ap.add_argument("--no_within", dest="within", action="store_false",
                    help="skip matching inside each page")
    ap.add_argument("--no_cross", dest="cross", action="store_false",
                    help="skip matching between pages")
    ap.add_argument("--max_overlay_pages", type=int, default=4,
                    help="write the all-pages overlay only up to this many pages")
    ap.add_argument("--top_k_within", type=int, default=CrossParams.top_k_within)
    ap.add_argument("--z_threshold", type=float, default=CrossParams.z_threshold)
    ap.add_argument("--margin", type=float, default=CrossParams.margin)
    ap.add_argument("--dtw_weight", type=float, default=CrossParams.dtw_weight,
                    help="0 = HOG only, 1 = DTW only (default 0.5)")
    ap.add_argument("--no_whiten", dest="whiten", action="store_false",
                    help="use plain mean-centred HOG instead of whitened HOG")
    ap.add_argument("--hog_weight", type=float, default=CrossParams.hog_weight)
    ap.add_argument("--hog_dtw_weight", type=float, default=CrossParams.hog_dtw_weight,
                    help="weight of HOG block-column sequences compared by DTW")
    ap.add_argument("--context_weight", type=float, default=CrossParams.context_weight,
                    help="z bonus per unit of neighbour support (0 = off)")
    ap.add_argument("--top_k", type=int, default=CrossParams.top_k)
    ap.add_argument("--size_shortlist", type=float, default=CrossParams.size_shortlist)
    a = ap.parse_args()

    cp = CrossParams(z_threshold=a.z_threshold, margin=a.margin, top_k=a.top_k,
                     size_shortlist=a.size_shortlist, dtw_weight=a.dtw_weight,
                     context_weight=a.context_weight, whiten=a.whiten,
                     hog_weight=a.hog_weight, hog_dtw_weight=a.hog_dtw_weight,
                     top_k_within=a.top_k_within)
    run_spotting(a.inputs, a.out_dir, cp, within=a.within, cross=a.cross,
                 max_overlay_pages=a.max_overlay_pages)


# --------------------------------------------------------------------------- #
# Run settings - used when the script is started without arguments
# (e.g. the Run button in PyCharm).
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent

SETTINGS = dict(
    images="/home/isuri/PycharmProjects/SETS_image_processing/data2",  # folder, file, or list
    out_dir=HERE / "result",              # where results are written
    within=True,                          # similar words inside each page
    cross=True,                           # similar words between every pair of pages
    max_overlay_pages=4,                  # all-pages overlay only for small runs
    params=CrossParams(
        z_threshold=3.25,                 # higher = fewer, more reliable matches
        whiten=True,                      # whitened HOG (best configuration)
        hog_weight=0.5, dtw_weight=0.5, hog_dtw_weight=0.0,
        context_weight=0.35,              # phrase-context bonus (0 = off)
    ),
)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:                 # arguments given -> command line
        main()
    else:                                 # no arguments -> SETTINGS above
        run_spotting(SETTINGS["images"], SETTINGS["out_dir"], SETTINGS["params"],
                     within=SETTINGS["within"], cross=SETTINGS["cross"],
                     max_overlay_pages=SETTINGS["max_overlay_pages"])