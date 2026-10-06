"""
Cross-document word spotting: which words on page A also occur on page B?

Reuses segment_manuscript.py, word_spotting.py, handwriting_norm.py and
profile_dtw.py (same folder).

Pipeline
--------
1. Segment both pages into tokens (segment_manuscript.segment).
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
   between the pages are reduced before any comparison.
4. Two complementary comparisons of every candidate pair:
   - HOG cosine on the normalised crop (shared mean-centring);
   - DTW distance between column-profile sequences (profile_dtw), which
     tolerates letters written wider or shifted sideways.
   Each is standardised over all pairs (global z) and the two are averaged
   into one score, weighted by `dtw_weight`.
5. Token-level scores: for each (A token, B token) the best combined score
   over their size-compatible candidate pairs (sizes normalised by line
   pitch).
6. Query-adaptive acceptance  [fix for the single global threshold]
   A pair (a, b) is a match if
     - it stands out from each word's own background: its z-score against
       all of a's scores AND against all of b's scores is >= `z_threshold`
       (background taken over all token pairs, before the size filter), and
     - its score is within `margin` (in global SD units) of the best score
       of query a AND of query b (relative to each word's own best, so
       words that naturally score lower are not penalised), and
     - a is in b's top-k and b is in a's top-k (reciprocal top-k; k = 3
       lets one word match up to three occurrences on the other page,
       while "hub" tokens similar to everything fail the reverse test), and
     - its score >= `score_floor` (global SD units).
7. Phrase context  [formulaic records]
   The registers repeat formulas ("Gio: Batta", "in casa di", "ambedue
   Inglesi"), so a real match usually has matching neighbours. For each
   pair (a, b) the support is the best z over neighbour pairs on the same
   side - (left(a), left(b)) or (right(a), right(b)) - taking neighbours
   up to `context_reach` tokens away in reading order (same column, across
   line breaks), so an extra brace or filler mark on one page is tolerated.
   Neighbour z-scores come from the unfiltered scores, so short formula
   words ("in", "di", "Gio:") count as evidence even though they are never
   reported as matches on their own. The ranking score is
       z_ctx = z + `context_weight` x clip(support, 0, `context_cap`)
   so supported pairs rise and may pass the threshold with a lower z; an
   unsupported pair is not penalised (names legitimately differ, e.g.
   "Giacomo Conti" vs "Giacomo Faccioli"). Acceptance uses z_ctx.

Outputs (in --out_dir)
----------------------
- cross_matches.csv : accepted matches (candidate ids, context z-score,
                      z-score, neighbour support, combined score, HOG cosine,
                      DTW distance)
- cross_matches.jpg : accepted matches side by side
- cross_overlay.jpg : both pages; each A word and its matches share a colour

Usage
-----
    python cross_spotting.py 000126.jpg mstest.jpg --out_dir cross
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from segment_manuscript import Box, Params, _runs, binarise, group_palette, segment
from handwriting_norm import NormParams, normalise_word, to_canvas
from profile_dtw import column_features, dtw_matrix
from core import SpotParams, compute_hog, tighten_to_ink


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
    z_threshold: float = 2.8          # min (context) z-score vs background
    margin: float = 0.75              # keep matches within margin of query's best
    top_k: int = 3                    # reciprocal top-k
    score_floor: float = 1.5          # min combined score (global SD units)
    # features
    dtw_weight: float = 0.5           # 0 = HOG only, 1 = DTW only
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
    is_full: bool = False             # True if the span is the whole token


@dataclass
class Page:
    name: str
    img: np.ndarray
    unit: float                       # line pitch in px
    cands: list[Candidate] = field(default_factory=list)


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
            is_full=full))
    return out


def load_page(image_path: str, cp: CrossParams, sp: SpotParams) -> Page:
    img, columns = segment(image_path, Params())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tokens = [(f"c{c}_l{l}_t{t}", b)
              for c, col in enumerate(columns, 1)
              for l, line in enumerate(col.lines, 1)
              for t, b in enumerate(line.tokens, 1)]
    unit = float(np.median([b.height for _, b in tokens]))
    page = Page(Path(image_path).stem, img, unit)
    for tid, b in tokens:
        page.cands += expand_token(gray, b, tid, unit, cp, sp)
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


def pair_scores(A: Page, B: Page, cp: CrossParams) -> dict[str, np.ndarray]:
    """Candidate-level matrices: HOG cosine, DTW distance and the combined
    score (weighted mean of the two, each globally standardised; DTW is
    negated so that higher = more similar for both)."""
    fa = np.stack([c.feat for c in A.cands]); fb = np.stack([c.feat for c in B.cands])
    cos = fa @ fb.T
    if cp.dtw_weight > 0:
        dtw = dtw_matrix(np.stack([c.seq for c in A.cands]),
                         np.stack([c.seq for c in B.cands]))
        score = (1 - cp.dtw_weight) * _gz(cos) + cp.dtw_weight * _gz(-dtw)
    else:
        dtw = np.full_like(cos, np.nan)
        score = _gz(cos)
    return dict(cos=cos, dtw=dtw, score=score)


def token_scores(A: Page, B: Page, S: np.ndarray, cp: CrossParams):
    """Token-level scores from the candidate-level score matrix S.

    T   : best score per (A token, B token) over size-compatible,
          word-length candidates
    U   : the same over ALL candidates, without size or width filters; used
          only as the background distribution for z-scores (the filters
          leave too few values, and short fragments are part of what a
          token must stand out from)
    arg : candidate indices behind each entry of T"""
    a_ids = list(dict.fromkeys(c.parent for c in A.cands))
    b_ids = list(dict.fromkeys(c.parent for c in B.cands))
    ra = np.array([a_ids.index(c.parent) for c in A.cands])
    rb = np.array([b_ids.index(c.parent) for c in B.cands])
    U, _ = _token_max(S, ra, rb, len(a_ids), len(b_ids))
    wa = np.array([c.box.width >= cp.min_match_lines * A.unit for c in A.cands])
    wb = np.array([c.box.width >= cp.min_match_lines * B.unit for c in B.cands])
    allowed = size_mask(A, B, cp) & wa[:, None] & wb[None, :]
    T, arg = _token_max(np.where(allowed, S, -np.inf), ra, rb, len(a_ids), len(b_ids))
    return T, U, arg


def _reading_neighbours(ids: list[str], d: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of the token d places before / after each token in reading
    order (-1 if none). `ids` is already in reading order (column, line,
    token); neighbours never cross a column boundary but do cross lines."""
    col = [t.split("_")[0] for t in ids]
    n = len(ids)
    left = np.array([k - d if k - d >= 0 and col[k - d] == col[k] else -1 for k in range(n)])
    right = np.array([k + d if k + d < n and col[k + d] == col[k] else -1 for k in range(n)])
    return left, right


def context_support(Zn: np.ndarray, a_ids: list[str], b_ids: list[str],
                    reach: int) -> np.ndarray:
    """support[p, q] = best Zn over neighbour pairs on the same side:
    (left_d1(p), left_d2(q)) or (right_d1(p), right_d2(q)) for d1, d2 in
    1..reach. Allowing unequal offsets tolerates an extra token on one page
    (a brace, a filler mark, a split word)."""
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
    P = pair_scores(A, B, cp)
    T, U, arg = token_scores(A, B, P["score"], cp)
    z = np.minimum((T - U.mean(1, keepdims=True)) / (U.std(1, keepdims=True) + 1e-8),
                   (T - U.mean(0, keepdims=True)) / (U.std(0, keepdims=True) + 1e-8))
    a_ids = list(dict.fromkeys(c.parent for c in A.cands))
    b_ids = list(dict.fromkeys(c.parent for c in B.cands))
    # Neighbour evidence uses the unfiltered scores U, so short formula words
    # ("in", "di", "Gio:", "loc.") can support a match even though they are
    # too unreliable to be reported as matches themselves.
    zn = np.minimum((U - U.mean(1, keepdims=True)) / (U.std(1, keepdims=True) + 1e-8),
                    (U - U.mean(0, keepdims=True)) / (U.std(0, keepdims=True) + 1e-8))
    support = context_support(zn, a_ids, b_ids, cp.context_reach)
    zc = z + cp.context_weight * np.clip(support, 0, cp.context_cap)

    k = cp.top_k
    top_a = np.argsort(-T, axis=1)[:, :k]                 # best B tokens per A
    top_b = np.argsort(-T, axis=0)[:k, :]                 # best A tokens per B
    best_a, best_b = T.max(axis=1), T.max(axis=0)

    pairs = []
    for p, q in zip(*np.nonzero((zc >= cp.z_threshold) & (T >= cp.score_floor))):
        s = T[p, q]
        if (q in top_a[p] and p in top_b[:, q]
                and s >= best_a[p] - cp.margin and s >= best_b[q] - cp.margin):
            i, j = arg[p, q]
            pairs.append(dict(i=int(i), j=int(j), z=float(zc[p, q]), z_base=float(z[p, q]),
                              support=float(max(support[p, q], 0.0)), score=float(s),
                              cos=float(P["cos"][i, j]), dtw=float(P["dtw"][i, j])))
    return sorted(pairs, key=lambda d: -d["z"])


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_a"); ap.add_argument("image_b")
    ap.add_argument("--out_dir", default="cross_out")
    ap.add_argument("--z_threshold", type=float, default=CrossParams.z_threshold)
    ap.add_argument("--margin", type=float, default=CrossParams.margin)
    ap.add_argument("--dtw_weight", type=float, default=CrossParams.dtw_weight,
                    help="0 = HOG only, 1 = DTW only (default 0.5)")
    ap.add_argument("--context_weight", type=float, default=CrossParams.context_weight,
                    help="z bonus per unit of neighbour support (0 = off)")
    ap.add_argument("--top_k", type=int, default=CrossParams.top_k)
    ap.add_argument("--size_shortlist", type=float, default=CrossParams.size_shortlist)
    a = ap.parse_args()

    cp = CrossParams(z_threshold=a.z_threshold, margin=a.margin, top_k=a.top_k,
                     size_shortlist=a.size_shortlist, dtw_weight=a.dtw_weight,
                     context_weight=a.context_weight)
    sp = SpotParams()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    A, B = load_page(a.image_a, cp, sp), load_page(a.image_b, cp, sp)
    compute_hog(A.cands + B.cands, sp)       # shared centring across pages
    pairs = match(A, B, cp)

    with open(out / "cross_matches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"{A.name}_candidate", f"{B.name}_candidate", "z_context",
                    "z_score", "neighbour_support", "combined_score", "hog_cosine",
                    "dtw_distance"])
        for p in pairs:
            w.writerow([A.cands[p["i"]].tid, B.cands[p["j"]].tid, f"{p['z']:.2f}",
                        f"{p['z_base']:.2f}", f"{p['support']:.2f}", f"{p['score']:.2f}",
                        f"{p['cos']:.3f}", f"{p['dtw']:.3f}"])
    cv2.imwrite(str(out / "cross_matches.jpg"), draw_pairs(A, B, pairs))
    cv2.imwrite(str(out / "cross_overlay.jpg"), draw_side_by_side(A, B, pairs),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"{A.name}: {len(A.cands)} candidates (line pitch {A.unit:.0f}px); "
          f"{B.name}: {len(B.cands)} candidates (line pitch {B.unit:.0f}px)")
    print(f"{len(pairs)} matches (z >= {cp.z_threshold}, margin {cp.margin}, "
          f"reciprocal top-{cp.top_k}, dtw_weight {cp.dtw_weight}, "
          f"context_weight {cp.context_weight})")
    print(f"{sum(p['z_base'] < cp.z_threshold for p in pairs)} of them admitted "
          f"only through phrase context")


if __name__ == "__main__":
    main()