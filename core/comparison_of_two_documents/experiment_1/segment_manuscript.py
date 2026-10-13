"""
Two-column manuscript segmentation: page -> columns -> lines -> tokens.

Pipeline
--------
1. Page extraction  : isolate the bright sheet from the dark scanning background.
2. Binarisation     : adaptive (Sauvola-style) thresholding of the ink.
3. Column detection : smoothed vertical projection profile; the gutter is the
                      deepest valley in the central band of the page.
4. Line detection   : smoothed horizontal projection per column; one line per
                      profile peak, boundaries at the minima between peaks.
5. Tokenisation     : vertical projection per line; a white run wider than an
                      adaptive threshold (inter-word gap) splits two tokens.
6. Size grouping    : tokens whose log-size vector (ln w, ln h) is within the
                      similarity threshold (default 95%, sim = exp(-Euclidean
                      distance)) of a group's reference box share a group id
                      and a colour.

Outputs (in --out_dir)
----------------------
- segmentation.json : nested boxes (column -> line -> token), full-image coords
- overlay.jpg       : visual check (columns blue, lines grey; only tokens in a
                      shared size group are drawn, one colour per group;
                      --show_singletons adds unmatched tokens in grey)
- tokens.csv        : one row per token with width, height and size group
- crops/            : optional token crops (--save_crops), e.g. for TrOCR

Usage
-----
    python segment_manuscript.py 000126.jpg --out_dir out --save_crops
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    """Axis-aligned box in full-image pixel coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int
    width: int = 0          # breadth (x1 - x0), filled automatically
    height: int = 0         # y1 - y0, filled automatically
    group: int = -1         # size-similarity group id (-1 = unassigned)

    def __post_init__(self):  # numpy ints -> native ints (JSON-safe)
        self.x0, self.y0, self.x1, self.y1 = map(int, (self.x0, self.y0, self.x1, self.y1))
        self.width, self.height = self.x1 - self.x0, self.y1 - self.y0


@dataclass
class Line:
    box: Box
    tokens: list[Box] = field(default_factory=list)


@dataclass
class Column:
    box: Box
    lines: list[Line] = field(default_factory=list)


@dataclass
class Params:
    """Tunable parameters. Lengths are fractions of page width/height so
    they transfer across scan resolutions."""
    page_margin: float = 0.015        # inner margin trimmed from the sheet edge
    gutter_band: tuple = (0.35, 0.65) # where to search for the column gutter
    col_smooth: float = 0.02          # smoothing window for the column profile
    line_smooth: float = 0.008        # smoothing window for the line profile
    min_line_gap: float = 0.018       # min distance between line centres
    line_prominence: float = 0.08     # peak prominence (fraction of max)
    min_line_ink: float = 0.002       # drop strips with less ink than this
    word_gap: float | None = None     # fixed inter-word gap (px); None = auto
    gap_factor: float = 1.8           # auto gap = factor * median letter gap
    min_token_w: float = 0.006        # drop tokens narrower than this


# --------------------------------------------------------------------------- #
# 1. Page extraction and binarisation
# --------------------------------------------------------------------------- #
def find_page(gray: np.ndarray, p: Params) -> Box:
    """Return the bounding box of the sheet (largest bright region)."""
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    mx, my = int(w * p.page_margin), int(h * p.page_margin)
    return Box(x + mx, y + my, x + w - mx, y + h - my)


def binarise(gray: np.ndarray) -> np.ndarray:
    """Ink = 1, paper = 0. Adaptive threshold copes with uneven paper tone."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    block = max(31, (min(gray.shape) // 40) | 1)          # odd block size
    ink = cv2.adaptiveThreshold(blur, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block, 15)
    # Remove speckle: drop tiny connected components.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(20, gray.size // 2_000_000)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


# --------------------------------------------------------------------------- #
# 2. Columns
# --------------------------------------------------------------------------- #
def split_columns(ink: np.ndarray, p: Params) -> list[tuple[int, int]]:
    """Return [(x0, x1), (x0, x1)] for the two columns (page-local coords)."""
    h, w = ink.shape
    prof = uniform_filter1d(ink.sum(axis=0).astype(float),
                            max(3, int(w * p.col_smooth)))
    lo, hi = int(w * p.gutter_band[0]), int(w * p.gutter_band[1])
    gutter = lo + int(np.argmin(prof[lo:hi]))
    return [(0, gutter), (gutter, w)]


# --------------------------------------------------------------------------- #
# 3. Lines
# --------------------------------------------------------------------------- #
def split_lines(ink: np.ndarray, p: Params, page_h: int) -> list[tuple[int, int]]:
    """Return [(y0, y1), ...] line strips within one column."""
    prof = uniform_filter1d(ink.sum(axis=1).astype(float),
                            max(3, int(page_h * p.line_smooth)))
    if prof.max() == 0:
        return []
    peaks, _ = find_peaks(prof,
                          distance=max(1, int(page_h * p.min_line_gap)),
                          prominence=prof.max() * p.line_prominence)
    if len(peaks) == 0:
        return []

    # Boundaries: profile minimum between consecutive peaks.
    cuts = [int(peaks[0] - (peaks[1] - peaks[0]) / 2) if len(peaks) > 1 else 0]
    for a, b in zip(peaks[:-1], peaks[1:]):
        cuts.append(a + int(np.argmin(prof[a:b])))
    cuts.append(int(peaks[-1] + (peaks[-1] - peaks[-2]) / 2) if len(peaks) > 1
                else ink.shape[0])
    cuts = np.clip(cuts, 0, ink.shape[0])

    min_ink = ink.shape[1] * page_h * p.min_line_ink * 0.1
    return [(int(y0), int(y1)) for y0, y1 in zip(cuts[:-1], cuts[1:])
            if ink[y0:y1].sum() > min_ink]


# --------------------------------------------------------------------------- #
# 4. Tokens
# --------------------------------------------------------------------------- #
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) of True runs in a 1-D boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def split_tokens(line_ink: np.ndarray, p: Params, page_w: int) -> list[tuple[int, int, int, int]]:
    """Return token boxes (x0, y0, x1, y1) in line-local coords, split on
    white gaps wider than the inter-word threshold."""
    # Use the core band (drop outer 15% top/bottom) so descenders/ascenders
    # from neighbouring lines do not bridge real word gaps.
    h = line_ink.shape[0]
    core = line_ink[int(0.15 * h): int(0.85 * h)]
    has_ink = core.sum(axis=0) > 0
    ink_runs = _runs(has_ink)
    if not ink_runs:
        return []

    gaps = np.array([b0 - a1 for (_, a1), (b0, _) in zip(ink_runs[:-1], ink_runs[1:])])
    if p.word_gap is not None:
        thr = p.word_gap
    elif len(gaps):
        thr = max(p.gap_factor * np.median(gaps), page_w * 0.006)
    else:
        thr = np.inf

    # Merge ink runs separated by gaps below the threshold.
    groups, (s, e) = [], ink_runs[0]
    for (ns, ne), g in zip(ink_runs[1:], gaps):
        if g < thr:
            e = ne
        else:
            groups.append((s, e)); s, e = ns, ne
    groups.append((s, e))

    tokens = []
    for x0, x1 in groups:
        if x1 - x0 < page_w * p.min_token_w:
            continue
        rows = np.flatnonzero(line_ink[:, x0:x1].any(axis=1))
        tokens.append((x0, int(rows[0]), x1, int(rows[-1]) + 1))
    return tokens


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def segment(image_path: str, p: Params = Params()) -> tuple[np.ndarray, list[Column]]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pg = find_page(gray, p)
    ink = binarise(gray[pg.y0:pg.y1, pg.x0:pg.x1])
    page_h, page_w = ink.shape

    columns: list[Column] = []
    for cx0, cx1 in split_columns(ink, p):
        col_ink = ink[:, cx0:cx1]
        col = Column(Box(pg.x0 + cx0, pg.y0, pg.x0 + cx1, pg.y1))
        for ly0, ly1 in split_lines(col_ink, p, page_h):
            line_ink = col_ink[ly0:ly1]
            line = Line(Box(pg.x0 + cx0, pg.y0 + ly0, pg.x0 + cx1, pg.y0 + ly1))
            for tx0, ty0, tx1, ty1 in split_tokens(line_ink, p, page_w):
                line.tokens.append(Box(pg.x0 + cx0 + tx0, pg.y0 + ly0 + ty0,
                                       pg.x0 + cx0 + tx1, pg.y0 + ly0 + ty1))
            if line.tokens:
                col.lines.append(line)
        columns.append(col)
    return img, columns


# --------------------------------------------------------------------------- #
# 5. Size-similarity grouping
# --------------------------------------------------------------------------- #
def ratio_similarity(a: Box, b: Box) -> float:
    """Weaker of the width ratio and height ratio (min/max), in [0, 1]."""
    return min(min(a.width, b.width) / max(a.width, b.width, 1),
               min(a.height, b.height) / max(a.height, b.height, 1))


def euclidean_similarity(a: Box, b: Box) -> float:
    """Scale-normalised Euclidean similarity of the size vectors (w, h):

        sim = 1 - ||(w1, h1) - (w2, h2)|| / max(||(w1, h1)||, ||(w2, h2)||)

    Dividing by the larger box's diagonal makes the score resolution-
    independent: 0.95 means the boxes' sizes differ by at most 5% of the
    larger box's diagonal length. Clipped to [0, 1]."""
    d = np.hypot(a.width - b.width, a.height - b.height)
    scale = max(np.hypot(a.width, a.height), np.hypot(b.width, b.height), 1.0)
    return float(max(0.0, 1.0 - d / scale))


def log_euclidean_similarity(a: Box, b: Box) -> float:
    """Euclidean distance between log-sizes, mapped to a similarity in (0, 1]:

        d   = sqrt( (ln w1 - ln w2)^2 + (ln h1 - ln h2)^2 )
        sim = exp(-d)

    Relative changes in width and height are weighted equally (unlike the
    diagonal-normalised version, which is dominated by width for word boxes),
    and the score is symmetric and resolution-independent. If only one
    dimension differs, sim equals that dimension's min/max ratio, so 0.95
    means roughly a 5% size difference."""
    d = np.hypot(np.log(max(a.width, 1) / max(b.width, 1)),
                 np.log(max(a.height, 1) / max(b.height, 1)))
    return float(np.exp(-d))


SIMILARITY_METRICS = {"log": log_euclidean_similarity,
                      "euclidean": euclidean_similarity,
                      "ratio": ratio_similarity}


def group_by_size(columns: list[Column], threshold: float = 0.95,
                  metric: str = "log") -> int:
    """Assign token.group so that every token in a group is >= `threshold`
    similar to the group's reference box under `metric` (a key of
    SIMILARITY_METRICS).

    Greedy leader clustering: tokens are visited largest-area first; each
    joins the most similar existing group whose leader it matches, otherwise
    it founds a new group. Comparing against a fixed leader (not chaining
    neighbour-to-neighbour) keeps groups from drifting in size.
    Returns the number of groups; ids are ordered by group size (0 = largest).
    """
    sim_fn = SIMILARITY_METRICS[metric]
    tokens = [t for c in columns for l in c.lines for t in l.tokens]
    leaders: list[Box] = []
    members: list[list[Box]] = []
    for tok in sorted(tokens, key=lambda b: b.width * b.height, reverse=True):
        sims = [sim_fn(tok, ld) for ld in leaders]
        best = int(np.argmax(sims)) if sims else -1
        if best >= 0 and sims[best] >= threshold:
            members[best].append(tok)
        else:
            leaders.append(tok); members.append([tok])

    # Renumber so the most populous group gets id 0.
    for gid, grp in enumerate(sorted(members, key=len, reverse=True)):
        for tok in grp:
            tok.group = gid
    return len(members)


def group_palette(n: int) -> list[tuple[int, int, int]]:
    """n distinct BGR colours. Hues step by the golden ratio so consecutive
    group ids never get near-identical colours; brightness alternates to
    separate groups that land on similar hues."""
    hsv = np.array([[[int(180 * ((i * 0.618034) % 1)), 255, 230 if i % 2 else 160]
                     for i in range(n)]], np.uint8)
    return [tuple(int(v) for v in c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]


def draw_overlay(img: np.ndarray, columns: list[Column],
                 show_singletons: bool = False) -> np.ndarray:
    """Columns blue, lines dark grey. Only tokens that share their size group
    with at least one other token are drawn, one colour per group. Tokens
    with no match are omitted (or drawn grey if `show_singletons`)."""
    tokens = [t for c in columns for l in c.lines for t in l.tokens]
    n_groups = max((t.group for t in tokens), default=-1) + 1
    counts = np.bincount([t.group for t in tokens if t.group >= 0], minlength=n_groups)
    # Spend the palette on shared groups only, so their colours stay distinct.
    shared = [g for g in range(n_groups) if counts[g] > 1]
    colours = dict(zip(shared, group_palette(len(shared))))
    out = img.copy()
    t = max(2, img.shape[1] // 1500)
    for col in columns:
        b = col.box
        cv2.rectangle(out, (b.x0, b.y0), (b.x1, b.y1), (255, 0, 0), t * 3)
        for i, line in enumerate(col.lines, 1):
            lb = line.box
            cv2.rectangle(out, (lb.x0, lb.y0), (lb.x1, lb.y1), (90, 90, 90), max(1, t // 2))
            cv2.putText(out, str(i), (lb.x0 + 5, lb.y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 90, 90), t)
            for tb in line.tokens:
                if tb.group in colours:
                    colour = colours[tb.group]
                elif show_singletons:
                    colour = (150, 150, 150)
                else:
                    continue                      # unmatched token: not drawn
                cv2.rectangle(out, (tb.x0, tb.y0), (tb.x1, tb.y1), colour, t * 2)
    return out


def save_outputs(img: np.ndarray, columns: list[Column], out_dir: str,
                 save_crops: bool = False, show_singletons: bool = False) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "segmentation.json", "w") as f:
        json.dump([asdict(c) for c in columns], f, indent=1)
    cv2.imwrite(str(out / "overlay.jpg"), draw_overlay(img, columns, show_singletons),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    # One row per token: position, size and group, for downstream analysis.
    with open(out / "tokens.csv", "w") as f:
        f.write("column,line,token,x0,y0,x1,y1,width,height,group\n")
        for c, col in enumerate(columns, 1):
            for l, line in enumerate(col.lines, 1):
                for t, b in enumerate(line.tokens, 1):
                    f.write(f"{c},{l},{t},{b.x0},{b.y0},{b.x1},{b.y1},"
                            f"{b.width},{b.height},{b.group}\n")
    if save_crops:
        crops = out / "crops"; crops.mkdir(exist_ok=True)
        for c, col in enumerate(columns, 1):
            for l, line in enumerate(col.lines, 1):
                for t, b in enumerate(line.tokens, 1):
                    cv2.imwrite(str(crops / f"c{c}_l{l:02d}_t{t:02d}.png"),
                                img[b.y0:b.y1, b.x0:b.x1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out_dir", default="seg_out")
    ap.add_argument("--save_crops", action="store_true")
    ap.add_argument("--word_gap", type=float, default=None,
                    help="fixed inter-word gap in px (default: adaptive)")
    ap.add_argument("--gap_factor", type=float, default=Params.gap_factor)
    ap.add_argument("--similarity", type=float, default=0.95,
                    help="min similarity for boxes to share a group/colour")
    ap.add_argument("--metric", choices=list(SIMILARITY_METRICS), default="log",
                    help="log (default): Euclidean distance of (ln w, ln h); "
                         "euclidean: diagonal-normalised distance of (w, h); "
                         "ratio: weaker of width/height min-max ratios")
    ap.add_argument("--show_singletons", action="store_true",
                    help="also draw unmatched tokens (grey) in the overlay")
    a = ap.parse_args()

    img, columns = segment(a.image, Params(word_gap=a.word_gap, gap_factor=a.gap_factor))
    n_groups = group_by_size(columns, a.similarity, a.metric)
    save_outputs(img, columns, a.out_dir, a.save_crops, a.show_singletons)
    tokens = [t for c in columns for l in c.lines for t in l.tokens]
    sizes = np.bincount([t.group for t in tokens])
    print(f"{n_groups} size groups at {a.similarity:.0%} {a.metric} similarity "
          f"({int((sizes > 1).sum())} shared, {int((sizes == 1).sum())} singletons)")
    for i, col in enumerate(columns, 1):
        print(f"Column {i}: {len(col.lines)} lines, "
              f"{sum(len(l.tokens) for l in col.lines)} tokens")


if __name__ == "__main__":
    main()