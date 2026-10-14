"""Curly-brace detection on a binary ink mask, without any trained model.

A brace in these registers is one of two things at pixel level:

  A. one or two narrow, loop-free ink components stacked vertically
     (the scribe often draws the upper and lower arm as separate strokes);
  B. a single long near-vertical stroke that touches neighbouring words, so it
     is not a component of its own and has to be traced with a path opening.

Both are then checked against the same context rules: the stroke stands in
white space, is not a straight slash or rule, and has text to its left on the
rows it spans (the items it groups).

All size thresholds are expressed in line spacings (LS) or stroke widths (SW)
so they follow the page scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from strokes import path_length


@dataclass
class Brace:
    """One detected brace. `mask` is a full-page boolean image of its pixels."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    mask: np.ndarray
    kind: str  # "stacked" (type A) or "traced" (type B)
    arms: int
    features: dict = field(default_factory=dict)

    @property
    def span(self) -> tuple[int, int]:
        """First and last row of the brace."""
        return self.bbox[1], self.bbox[1] + self.bbox[3] - 1

    def right_edge(self, y0: int, y1: int) -> int:
        """Right-most brace pixel in rows [y0, y1], or the bbox edge if none."""
        cols = np.where(self.mask[max(0, y0) : y1 + 1].any(axis=0))[0]
        return int(cols.max()) if len(cols) else self.bbox[0] + self.bbox[2] - 1


# --------------------------------------------------------------------------
# feature helpers
# --------------------------------------------------------------------------
def _row_extents(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """(y, xmin, xmax) for every row of `mask` that has pixels."""
    ys, xs = np.where(mask)
    out = []
    for y in np.unique(ys):
        r = xs[ys == y]
        out.append((int(y), int(r.min()), int(r.max())))
    return out


def stroke_features(mask: np.ndarray, ink: np.ndarray, ls: int, sw: float) -> dict:
    """Shape and context measurements for a candidate stroke mask."""
    rows = _row_extents(mask)
    ys = np.array([r[0] for r in rows])
    cx = np.array([(r[1] + r[2]) / 2 for r in rows])
    slope, icpt = np.polyfit(ys, cx, 1) if len(rows) > 2 else (0.0, cx.mean())
    resid = float(np.sqrt(np.mean((slope * ys + icpt - cx) ** 2)))

    # isolation: share of rows with a clear gap of `g` pixels beside the stroke
    g, d = int(0.3 * ls), int(round(sw))
    w = ink.shape[1]
    other = ink & ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 2 * d + 1), np.uint8)).astype(bool)
    iso_l = np.mean([not other[y, max(0, a - g) : a].any() for y, a, _ in rows])
    iso_r = np.mean([not other[y, b + 1 : min(w, b + 1 + g)].any() for y, _, b in rows])

    # left context: rows in the span that have text within 6 LS to the left
    x0 = min(r[1] for r in rows)
    y0, y1 = int(ys.min()), int(ys.max())
    left = other[y0 : y1 + 1, max(0, x0 - 6 * ls) : max(0, x0 - g)]
    right = other[y0 : y1 + 1, max(r[2] for r in rows) + g : max(r[2] for r in rows) + 6 * ls]
    return {
        "height_ls": round((y1 - y0 + 1) / ls, 2),
        "slope": round(float(slope), 2),
        "resid_sw": round(resid / sw, 2),
        "iso_left": round(float(iso_l), 2),
        "iso_right": round(float(iso_r), 2),
        "left_text": round(float(left.any(axis=1).mean()) if left.size else 0.0, 2),
        "right_text": round(float(right.any(axis=1).mean()) if right.size else 0.0, 2),
    }


def _has_hole(comp: np.ndarray) -> bool:
    _, hier = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    return hier is not None and any(h[3] != -1 for h in hier[0])


def _end_x(comp_rows: list[tuple[int, int, int]], top: bool, frac: float = 0.25) -> float:
    """Mean centre x of the top or bottom `frac` of a stroke."""
    k = max(1, int(len(comp_rows) * frac))
    part = comp_rows[:k] if top else comp_rows[-k:]
    return float(np.mean([(a + b) / 2 for _, a, b in part]))


# --------------------------------------------------------------------------
# type A: stacked isolated components
# --------------------------------------------------------------------------
def _arm_primitives(ink: np.ndarray, ls: int) -> list[dict]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    prims = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if not (h >= 0.4 * ls and w <= 0.9 * ls and h >= 1.3 * w):
            continue
        comp = lab[y : y + h, x : x + w] == i
        if _has_hole(comp):
            continue
        full = np.zeros_like(ink)
        full[y : y + h, x : x + w] = comp
        rows = _row_extents(full)
        prims.append({"mask": full, "y0": y, "y1": y + h - 1, "rows": rows,
                      "top_x": _end_x(rows, True), "bot_x": _end_x(rows, False)})
    return sorted(prims, key=lambda p: p["y0"])


def _stands_alone(p: dict, ink: np.ndarray, ls: int, sw: float) -> bool:
    """True if a primitive has white space on both sides (cached on the dict)."""
    if "alone" not in p:
        f = stroke_features(p["mask"], ink, ls, sw)
        p["alone"] = f["iso_left"] >= 0.6 and f["iso_right"] >= 0.6
    return p["alone"]


def _chain(prims: list[dict], ink: np.ndarray, ls: int, sw: float) -> list[list[dict]]:
    """Greedily link each primitive to the nearest one starting just below it.

    An arm is only appended if it stands alone; this stops a chain from running
    on into a letter of the next text line.
    """
    used, chains = set(), []
    for i, p in enumerate(prims):
        if i in used:
            continue
        chain, cur = [p], p
        used.add(i)
        while True:
            best, best_gap = None, None
            for j, q in enumerate(prims):
                if j in used:
                    continue
                gap = q["y0"] - cur["y1"]
                if -0.15 * ls <= gap <= 0.6 * ls and abs(q["top_x"] - cur["bot_x"]) <= 0.4 * ls:
                    if (best is None or gap < best_gap) and _stands_alone(q, ink, ls, sw):
                        best, best_gap = j, gap
            if best is None:
                break
            used.add(best)
            cur = prims[best]
            chain.append(cur)
        chains.append(chain)
    return chains


# --------------------------------------------------------------------------
# type B: long strokes traced through touching text
# --------------------------------------------------------------------------
def _traced_candidates(ink: np.ndarray, ls: int, min_ls: float = 1.6) -> list[np.ndarray]:
    long_px = path_length(ink) >= min_ls * ls
    n, lab = cv2.connectedComponents(long_px.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n)]


def _grow(seed: np.ndarray, ink: np.ndarray, ls: int) -> np.ndarray:
    """Extend a traced stroke over the rest of the brace.

    The vertical path opening cannot follow the sideways jog at the cusp, so a
    traced brace comes back as its longer arm only. Whatever else hangs on the
    same ink component is added if it is narrow and stays in the stroke's own
    x-band; words that merely touch the brace are wide and are left out.
    """
    n, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[seed & ink])
    comp = np.isin(lab, ids[ids > 0])
    xs = np.where(seed.any(axis=0))[0]
    x_lo, x_hi = xs.min() - 0.5 * ls, xs.max() + 0.5 * ls
    out = seed & ink
    m, rest, stats, _ = cv2.connectedComponentsWithStats((comp & ~seed).astype(np.uint8), connectivity=8)
    for i in range(1, m):
        x, _, w, _, _ = stats[i]
        if w <= 0.9 * ls and x >= x_lo and x + w <= x_hi:
            out |= rest == i
    return out


def _merge_touching(masks: list[np.ndarray], ls: int) -> list[list[int]]:
    """Group masks that overlap once stretched vertically by 0.2 LS."""
    k = np.ones((int(0.4 * ls) | 1, 5), np.uint8)
    fat = [cv2.dilate(m.astype(np.uint8), k).astype(bool) for m in masks]
    parent = list(range(len(masks)))

    def find(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if (fat[i] & fat[j]).any():
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# --------------------------------------------------------------------------
# refinement of an accepted brace
# --------------------------------------------------------------------------
def _absorb_fragments(mask: np.ndarray, ink: np.ndarray, ls: int, sw: float) -> np.ndarray:
    """Add small detached pieces that continue the brace at either end.

    Faint ink breaks a brace into fragments that are too short to count as an
    arm on their own. A fragment is absorbed if it is small, loop-free, lies
    within 0.25 LS of the current end, lines up with it, and stands in white
    space (so dots and letters of the next text line are not picked up).
    """
    n, lab, stats, cents = cv2.connectedComponentsWithStats((ink & ~mask).astype(np.uint8), connectivity=8)
    small = [i for i in range(1, n)
             if stats[i, cv2.CC_STAT_HEIGHT] <= 0.6 * ls and stats[i, cv2.CC_STAT_WIDTH] <= 0.5 * ls
             and stats[i, cv2.CC_STAT_AREA] >= 5 * sw * sw]
    out, changed = mask.copy(), True
    while changed:
        changed = False
        rows = _row_extents(out)
        ends = {"top": (rows[0][0], _end_x(rows, True)), "bottom": (rows[-1][0], _end_x(rows, False))}
        for i in list(small):
            x, y, w, h, _ = stats[i]
            gap_top, gap_bot = ends["top"][0] - (y + h - 1), y - ends["bottom"][0]
            if 0 <= gap_bot <= 0.25 * ls:
                ref_x = ends["bottom"][1]
            elif 0 <= gap_top <= 0.25 * ls:
                ref_x = ends["top"][1]
            else:
                continue
            if abs(cents[i][0] - ref_x) > 0.3 * ls:
                continue
            piece = lab == i
            if _has_hole(piece[y : y + h, x : x + w]):
                continue
            f = stroke_features(piece, ink & ~out, ls, sw)
            if f["iso_left"] >= 0.6 and f["iso_right"] >= 0.6:
                out |= piece
                small.remove(i)
                changed = True
    return out


def _absorb_inner(mask: np.ndarray, ink: np.ndarray, ls: int, sw: float) -> np.ndarray:
    """Add small detached strokes that lie between the arms of the brace.

    When the scribe draws one small brace per line, the lower half of the
    upper brace is a short separate stroke in the middle of the detection.
    It is absorbed if it is small, loop-free, inside the brace's rows, within
    0.35 LS of the brace's own centre line at that height (so a comma after
    the last word is left alone), and stands in white space on both sides.
    """
    rows = _row_extents(mask)
    row_y = np.array([r[0] for r in rows])
    row_x = np.array([(r[1] + r[2]) / 2 for r in rows])
    out = mask.copy()
    n, lab, stats, cents = cv2.connectedComponentsWithStats((ink & ~mask).astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (y >= row_y[0] and y + h - 1 <= row_y[-1]):
            continue
        if area < 2.5 * sw * sw or max(w, h) > 0.9 * ls:
            continue
        if abs(cents[i][0] - np.interp(cents[i][1], row_y, row_x)) > 0.35 * ls:
            continue
        piece = lab == i
        if _has_hole(piece[y : y + h, x : x + w]):
            continue
        f = stroke_features(piece, ink & ~out, ls, sw)
        if f["iso_left"] >= 0.6 and f["iso_right"] >= 0.6:
            out |= piece
    return out


def _absorb_stubs(mask: np.ndarray, ink: np.ndarray, ls: int, sw: float) -> np.ndarray:
    """Add small loop-free pieces of ink that touch the brace.

    The tracing leaves out the tip of a tail that swings far to one side.
    What remains is a short open stroke attached to the brace; a letter that
    touches the brace is instead attached to its word, which is much larger.
    """
    near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((ink & ~mask).astype(np.uint8), connectivity=8)
    out = mask.copy()
    for i in np.unique(lab[near & (lab > 0)]):
        x, y, w, h, area = stats[i]
        if area <= 12 * sw * sw and max(w, h) <= ls and not _has_hole(lab[y : y + h, x : x + w] == i):
            out |= lab == i
    return out


def _remove_end_loops(mask: np.ndarray, sw: float, end_frac: float = 0.2) -> np.ndarray:
    """Drop closed loops at the ends of a brace: they are letters it touches.

    A brace arm is an open stroke. When its tip touches an "o" or a looped
    "p", the path tracing follows the stroke into that letter. Any enclosed
    hole in the top or bottom `end_frac` of the mask marks such a letter; the
    ring around the hole is removed together with whatever it cut off.
    """
    ys = np.where(mask.any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    holes = binary_fill_holes(mask) & ~mask
    n, lab, stats, cents = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    bad = np.zeros_like(mask)
    for i in range(1, n):
        cy = cents[i][1]
        at_end = cy <= y0 + end_frac * (y1 - y0) or cy >= y1 - end_frac * (y1 - y0)
        if stats[i, cv2.CC_STAT_AREA] >= 0.5 * sw * sw and at_end:
            bad |= lab == i
    if not bad.any():
        return mask
    r = int(np.ceil(1.5 * sw))
    ring = cv2.dilate(bad.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    cut = mask & ~ring.astype(bool)
    # of every original piece keep only the largest part that survived the cut
    out = np.zeros_like(mask)
    n0, lab0 = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n0):
        m, sub, st, _ = cv2.connectedComponentsWithStats((cut & (lab0 == i)).astype(np.uint8), connectivity=8)
        if m > 1:
            out |= sub == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return out if out.sum() >= 0.6 * mask.sum() else mask


def _trim_embedded(mask: np.ndarray, ink: np.ndarray, ls: int, sw: float, max_frac: float = 0.35) -> np.ndarray:
    """Cut an end of the brace that runs on into a letter of another text line.

    When a brace arm ends on a letter stem (e.g. a "t" in the line below) the
    two form one stroke. Those rows are recognisable because ink sits within
    a letter gap (0.25 LS) on *both* sides, which never happens at a free brace
    end. Rows are removed from each end while that holds.
    """
    rows = _row_extents(mask)
    g, d = int(0.25 * ls), int(round(sw))
    other = ink & ~cv2.dilate(mask.astype(np.uint8), np.ones((3, 2 * d + 1), np.uint8)).astype(bool)
    w = ink.shape[1]
    emb = [bool(other[y, max(0, a - g) : a].any() and other[y, b + 1 : min(w, b + 1 + g)].any())
           for y, a, b in rows]
    limit, tol = int(max_frac * len(rows)), max(2, int(0.08 * ls))

    def run(seq: list[bool]) -> int:
        """Rows to drop from the start of `seq`, bridging gaps shorter than `tol`."""
        cut, miss = 0, 0
        for k, e in enumerate(seq[:limit]):
            if e:
                cut, miss = k + 1, 0
            else:
                miss += 1
                if miss > tol:
                    break
        return cut

    top, bot = run(emb), run(emb[::-1])
    out = mask.copy()
    if top:
        out[: rows[top][0]] = False
    if bot:
        out[rows[-bot - 1][0] + 1 :] = False
    return out


# --------------------------------------------------------------------------
# decision
# --------------------------------------------------------------------------
def is_brace(f: dict, kind: str, arms: int = 1) -> bool:
    """Context rules shared by both candidate types.

    A single loop-free stroke taller than 1.6 lines can hardly be a letter, so
    it only needs moderate isolation. Shorter strokes must be stacked (two or
    more arms) and stand clearly in white space on both sides, otherwise
    ascenders and descenders of neighbouring lines pass as braces.
    """
    if abs(f["slope"]) > 0.45:  # leaning: a slash, not a brace
        return False
    if f["resid_sw"] < 0.35:  # dead straight: a rule or fold shadow
        return False
    if f["left_text"] < 0.4:  # nothing to its left to group
        return False
    if kind == "stacked" and arms > 1:
        return f["height_ls"] >= 1.1 and f["iso_left"] >= 0.75 and f["iso_right"] >= 0.75
    return f["height_ls"] >= 1.6 and f["iso_left"] >= 0.5 and f["iso_right"] >= 0.6


def detect_braces(ink: np.ndarray, ls: int, sw: float, debug: list | None = None) -> list[Brace]:
    """Return all braces found in the ink mask."""
    found: list[Brace] = []
    taken = np.zeros_like(ink)

    for chain in _chain(_arm_primitives(ink, ls), ink, ls, sw):
        mask = np.zeros_like(ink)
        for p in chain:
            mask |= p["mask"]
        f = stroke_features(mask, ink, ls, sw)
        ok = is_brace(f, "stacked", len(chain))
        if debug is not None and f["height_ls"] >= 1.1:
            debug.append(("stacked", ok, f, mask, len(chain)))
        if ok:
            found.append(_make(mask, "stacked", len(chain), f))
            taken |= mask

    for mask in _traced_candidates(ink, ls):
        if (mask & taken).sum() > 0.3 * mask.sum():
            continue  # already found as a stacked brace
        f = stroke_features(mask, ink, ls, sw)
        ok = is_brace(f, "traced")
        if debug is not None:
            debug.append(("traced", ok, f, mask, 1))
        if ok:
            found.append(_make(_grow(mask, ink, ls), "traced", 1, f))

    # two halves of one brace (split at the cusp) become one detection
    merged: list[Brace] = []
    for group in _merge_touching([b.mask for b in found], ls):
        if len(group) == 1:
            merged.append(found[group[0]])
            continue
        mask = np.zeros_like(ink)
        for i in group:
            mask |= found[i].mask
        first = found[group[0]]
        merged.append(_make(mask, first.kind, sum(found[i].arms for i in group),
                            stroke_features(mask, ink, ls, sw)))
    refined = []
    for b in merged:
        mask = _absorb_inner(_absorb_fragments(b.mask, ink, ls, sw), ink, ls, sw)
        mask = _absorb_stubs(mask, ink, ls, sw)
        mask = _trim_embedded(_remove_end_loops(mask, sw), ink, ls, sw)
        refined.append(_make(mask, b.kind, b.arms, stroke_features(mask, ink, ls, sw)))
    return sorted(refined, key=lambda b: (b.bbox[0] > ink.shape[1] // 2, b.bbox[1]))


def _make(mask: np.ndarray, kind: str, arms: int, f: dict) -> Brace:
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return Brace(bbox, mask, kind, arms, f)