"""Line polygons from Kraken (the eScriptorium segmenter) + bracket grouping.

Kraken finds each line's baseline and its smooth outline (boundary). This script then
uses the rule-based bracket detector from page_polygons.py to group the bracketed lines
and attach each bracket's label lines.

Usage
  python kraken_lines.py                     # page.png next to this script
  python kraken_lines.py scan.jpg --xml      # also write PAGE-XML for eScriptorium
or inline:
  from kraken_lines import run
  res = run("page.png")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import page_polygons as pp                       # ink mask, columns, bracket detection

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "page.png"
DEFAULT_OUT = SCRIPT_DIR / "out_kraken"


@dataclass
class KLine:
    id: str
    baseline: np.ndarray                         # (N, 2)
    polygon: np.ndarray                          # (M, 2)
    column: int = -1

    @property
    def x_start(self) -> float: return float(self.baseline[:, 0].min())
    @property
    def x_end(self) -> float: return float(self.baseline[:, 0].max())
    @property
    def y_mid(self) -> float: return float(np.median(self.baseline[:, 1]))


@dataclass
class Group:
    id: str
    bracket: pp.Comp
    column: int
    lines: list[str] = field(default_factory=list)
    label_lines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 1. Kraken
def kraken_segment(img: Image.Image, model_path: str | None = None) -> list[KLine]:
    """Run Kraken's baseline segmenter (default model: the one eScriptorium ships)."""
    from kraken import blla
    from kraken.lib import vgsl

    model = vgsl.TorchVGSLModel.load_model(model_path) if model_path else None
    seg = blla.segment(img, model=model) if model else blla.segment(img)
    lines = []
    for i, ln in enumerate(seg.lines, 1):
        if not ln.baseline or not ln.boundary:
            continue
        lines.append(KLine(f"l{i:03d}", np.array(ln.baseline, float), np.array(ln.boundary, float)))
    return lines


def load_pagexml(xml_path: str | Path) -> list[KLine]:
    """Lines (baseline + outline) from a PAGE-XML file, e.g. exported from eScriptorium.

    Use this to work with exactly the segmentation you see (and corrected) in eScriptorium.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    ns = {"p": root.tag.split("}")[0].strip("{")}
    pts = lambda s: np.array([[float(v) for v in p.split(",")] for p in s.split()], float)
    lines = []
    for i, tl in enumerate(root.iter(f"{{{ns['p']}}}TextLine"), 1):
        coords, base = tl.find("p:Coords", ns), tl.find("p:Baseline", ns)
        if coords is None or base is None:
            continue
        lines.append(KLine(tl.get("id") or f"l{i:03d}", pts(base.get("points")), pts(coords.get("points"))))
    return lines


def write_pagexml(img: Image.Image, image_name: str, out_path: Path) -> None:
    """PAGE-XML that eScriptorium / Transkribus can import (segmentation only)."""
    from kraken import blla, serialization
    seg = blla.segment(img)
    out_path.write_text(serialization.serialize(seg, image_size=img.size, template="pagexml"))


# ---------------------------------------------------------------- 1b. absorb prefixes
def extend_baselines_left(lines: list[KLine], ink: np.ndarray, char_h: float,
                          reach: float = 2.5, above: float = 1.6, below: float = 0.5,
                          max_steps: int = 5) -> int:
    """Extend a baseline leftwards over ink that clearly belongs to the line's start.

    Raised, slanted prefixes such as "Sl" (Signori) are often left out of Kraken's baseline,
    so they end up outside the line outline. A component is absorbed when it
      * ends within `reach` letter heights left of the baseline's start,
      * lies vertically around the baseline (from `above` x char height above it
        to `below` x char height under it), and
      * is not inside another line's outline, and
      * is not taller than ~2.5 letters (so a label never extends over its bracket).
    Repeated so chains like "Sl" + "c" are picked up. Returns the number of extended lines.
    """
    lab, st, ids = pp.components(ink)
    others = []
    for l in lines:
        m = np.zeros(ink.shape, np.uint8)
        cv2.fillPoly(m, [l.polygon.astype(np.int32)], 1)
        others.append(m.astype(bool))
    cx = st[:, 0] + st[:, 2] / 2
    cy = st[:, 1] + st[:, 3] / 2
    n_ext = 0
    for li, l in enumerate(lines):
        bl = l.baseline[np.argsort(l.baseline[:, 0])]
        x_start, y_start = bl[0]
        extended = False
        for _ in range(max_steps):
            right = st[:, 0] + st[:, 2]
            sel = ((right <= x_start + 0.3 * char_h) & (right >= x_start - reach * char_h)
                   & (cy >= y_start - above * char_h) & (cy <= y_start + below * char_h)
                   & (st[:, 4] >= 0.1 * char_h ** 2)
                   & (st[:, 3] <= 2.5 * char_h))                     # never swallow a bracket
            cand = []
            for k in np.nonzero(sel)[0]:
                px, py = int(cx[k]), int(cy[k])
                if any(o[py, px] for j, o in enumerate(others) if j != li):
                    continue                                      # belongs to another line
                cand.append(k)
            if not cand:
                break
            new_x = float(st[cand, 0].min())
            if new_x >= x_start - 1:
                break
            x_start = new_x
            extended = True
        if extended:
            l.baseline = np.vstack([[x_start, y_start], bl])
            n_ext += 1
    return n_ext


def recompute_boundaries(img: Image.Image, lines: list[KLine]) -> None:
    """Redraw every outline from (possibly edited) baselines with Kraken's own polygonizer."""
    from kraken.lib.segmentation import calculate_polygonal_environment

    bls = [[tuple(map(int, p)) for p in l.baseline] for l in lines]
    polys = calculate_polygonal_environment(img.convert("L"), bls)
    for l, p in zip(lines, polys):
        if p is not None and len(p) >= 3:
            l.polygon = np.array(p, float)


# ---------------------------------------------------------------- 1c. tight, non-overlapping outlines
def tighten_boundaries(lines: list[KLine], shape: tuple[int, int], char_h: float,
                       ink: np.ndarray | None = None, up: float = 1.5, down: float = 0.7,
                       simplify: float = 1.5, max_letter: float = 4.5,
                       split_share: float = 0.2) -> None:
    """eScriptorium-style outlines: tight around each baseline and never overlapping.

    * each line keeps only pixels of its Kraken outline that lie within `up` letter heights
      above and `down` letter heights below its baseline;
    * where outlines still overlap, every pixel goes to the line whose baseline is nearest;
    * whole letters: every ink stroke goes, complete and slightly padded, to the line that
      already holds most of it (or, if no line holds any of it, to the line with the nearest
      baseline), so tall or raised letters such as "Sl" are never cut or left out
      (strokes taller than `max_letter` letter heights, e.g. brackets, are left alone);
    * the result is traced back into a polygon (largest piece kept).
    """
    H, W = shape
    best_d = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), -1, np.int32)
    for i, l in enumerate(lines):
        x0, y0, w, h = cv2.boundingRect(l.polygon.astype(np.int32))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x0 + w + 1), min(H, y0 + h + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(sub, [(l.polygon - [x0, y0]).astype(np.int32)], 1)
        # signed vertical offset from the baseline in every column of the box
        bl = l.baseline[np.argsort(l.baseline[:, 0])]
        xs = np.arange(x0, x1)
        by = np.interp(xs, bl[:, 0], bl[:, 1])
        dy = np.arange(y0, y1)[:, None] - by[None, :]            # >0 below the baseline
        band = (dy >= -up * char_h) & (dy <= down * char_h)
        # baseline distance (outside the baseline's x-range, add the horizontal gap)
        gap_x = np.maximum(0, np.maximum(bl[0, 0] - xs, xs - bl[-1, 0]))[None, :]
        d = np.hypot(np.where(dy < 0, dy / up, dy / down), gap_x / char_h).astype(np.float32)
        m = (sub > 0) & band
        cur = best_d[y0:y1, x0:x1]
        win = m & (d < cur)
        cur[win] = d[win]
        owner[y0:y1, x0:x1][win] = i
    if ink is not None:
        nearest_max = 1.6                                   # in band units (1 = band edge)
        base = []
        for l in lines:
            b = l.baseline[np.argsort(l.baseline[:, 0])]
            base.append((b[:, 0], b[:, 1]))
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        pad = max(2, int(0.15 * char_h))
        kern = np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if h > max_letter * char_h or a < 0.05 * char_h ** 2:
                continue
            comp = lab[y:y + h, x:x + w] == k
            votes = owner[y:y + h, x:x + w][comp]
            votes = votes[votes >= 0]
            if votes.size:
                counts = np.bincount(votes)
                top2 = np.argsort(counts)[::-1][:2]
                if len(top2) == 2 and counts[top2[1]] >= split_share * votes.size:
                    # Touching strokes from two lines (e.g. the "l" of "Sl" joined to the "C"
                    # of the line above): split pixel-wise by the nearer baseline instead of
                    # giving the whole stroke to one line.
                    ys, xs = np.nonzero(comp)
                    ys, xs = ys + y, xs + x
                    dist = []
                    for j in top2:
                        bx, byy = base[j]
                        dyj = ys - np.interp(xs, bx, byy)
                        dist.append(np.where(dyj < 0, -dyj / (up * char_h), dyj / (down * char_h)))
                    pick = top2[np.argmin(np.stack(dist), axis=0)]
                    for j in top2:
                        sel = pick == j
                        if not sel.any():
                            continue
                        yy0, xx0 = max(0, ys[sel].min() - pad), max(0, xs[sel].min() - pad)
                        yy1, xx1 = min(H, ys[sel].max() + 1 + pad), min(W, xs[sel].max() + 1 + pad)
                        g = np.zeros((yy1 - yy0, xx1 - xx0), np.uint8)
                        g[ys[sel] - yy0, xs[sel] - xx0] = 1
                        g = cv2.dilate(g, kern).astype(bool)
                        owner[yy0:yy1, xx0:xx1][g] = int(j)
                    continue
                win = int(counts.argmax())
            else:
                # Stroke outside every band (e.g. the raised "l" of "Sl"): give it to the line
                # whose baseline is nearest, measured like the band (letters sit above it).
                cx, cy = x + w / 2, y + h / 2
                best, best_d = -1, nearest_max
                for i, (bx, byy) in enumerate(base):
                    if not (bx[0] - 2 * char_h <= cx <= bx[-1] + char_h):
                        continue
                    dy = cy - float(np.interp(cx, bx, byy))
                    dd = -dy / (up * char_h) if dy < 0 else dy / (down * char_h)
                    if dd < best_d:
                        best, best_d = i, dd
                if best < 0:
                    continue
                win = best
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(H, y + h + pad), min(W, x + w + pad)
            grown = np.zeros((y1 - y0, x1 - x0), np.uint8)
            grown[y - y0:y - y0 + h, x - x0:x - x0 + w] = comp
            grown = cv2.dilate(grown, kern).astype(bool)
            owner[y0:y1, x0:x1][grown] = win
    for i, l in enumerate(lines):
        m = (owner == i).astype(np.uint8)
        if not m.any():
            continue
        # bridge the small gaps between a line's band and its whole letters
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((max(3, int(0.3 * char_h)),) * 2, np.uint8))
        m &= ((owner == i) | (owner == -1)).astype(np.uint8)         # never into another line
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        if len(cnts) > 1:
            # Separate islands (a raised letter that touches the band nowhere) are joined to
            # the main body by a narrow bridge, so the line stays one polygon.
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            body = cnts[0].reshape(-1, 2)
            bw = max(3, int(0.35 * char_h))
            for c in cnts[1:]:
                if cv2.contourArea(c) < 0.05 * char_h ** 2:
                    continue
                pts = c.reshape(-1, 2)
                d = np.linalg.norm(pts[:, None, :].astype(float) - body[None, ::4, :], axis=2)
                a, b = np.unravel_index(d.argmin(), d.shape)
                if d[a, b] > 3 * char_h:
                    continue
                cv2.line(m, tuple(int(v) for v in pts[a]), tuple(int(v) for v in body[::4][b]), 1, bw)
            m &= ((owner == i) | (owner == -1)).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, simplify, True).reshape(-1, 2)
        if len(c) >= 3:
            l.polygon = c.astype(float)


# ---------------------------------------------------------------- 2. brackets
def clip_line(line: KLine, x_lo: float, x_hi: float, new_id: str) -> KLine | None:
    """Part of a line between x_lo and x_hi (polygon clipped, baseline cut and interpolated)."""
    from shapely.geometry import Polygon, box

    poly = Polygon(line.polygon).buffer(0)
    y0, y1 = poly.bounds[1] - 1, poly.bounds[3] + 1
    part = poly.intersection(box(x_lo, y0, x_hi, y1))
    if part.is_empty:
        return None
    if part.geom_type != "Polygon":                                  # keep the largest piece
        part = max(getattr(part, "geoms", [part]), key=lambda g: g.area)
        if part.geom_type != "Polygon":
            return None
    bl = line.baseline
    xs = np.clip(np.r_[x_lo, bl[:, 0], x_hi], bl[:, 0].min(), bl[:, 0].max())
    xs = np.unique(xs[(xs >= x_lo) & (xs <= x_hi)])
    if xs.size < 2:
        return None
    order = np.argsort(bl[:, 0])
    ys = np.interp(xs, bl[order, 0], bl[order, 1])
    return KLine(new_id, np.c_[xs, ys], np.array(part.exterior.coords), line.column)


def split_across_brackets(lines: list[KLine], brackets: list[tuple[int, pp.Comp]],
                          pitch: dict, char_h: float) -> list[KLine]:
    """Kraken may run a line through a bracket into its label ("Pietro Celè } ambedue").
    Cut such lines at the bracket: left part = bracketed text, right part = label text."""
    out = []
    for l in lines:
        hit = None
        for ci, br in brackets:
            p = pitch[ci]
            if (l.column == ci and br.y1 <= l.y_mid <= br.y2 + 0.25 * p
                    and l.x_start < br.x1 - char_h and l.x_end > br.x2 + char_h):
                hit = br
                break
        if hit is None:
            out.append(l)
            continue
        left = clip_line(l, l.x_start - 2 * char_h, hit.x1 + 0.3 * char_h, l.id + "a")
        right = clip_line(l, hit.x2, l.x_end + 2 * char_h, l.id + "b")
        out.extend(p for p in (left, right) if p is not None)
    return out



def bracket_groups(rgb: np.ndarray, lines: list[KLine], align_tol: float = 0.35,
                   core: float = 0.6, label_reach: float = 6.0) -> tuple[list[Group], dict]:
    """Detect brackets (rule-based) and assign Kraken lines to them.

    members : same column, baseline height within the bracket's span, ending left of
              the bracket, and starting near the column's left margin (drops "23 Mag.").
    label   : lines starting just right of the bracket whose mid-height lies in the
              bracket's central part (keeps "partirono disse" at its tip out).
    """
    raw = pp.ink_mask(rgb)
    char_h = pp.estimate_char_height(raw)
    paper = cv2.erode(pp.page_mask(rgb).astype(np.uint8), np.ones((int(char_h),) * 2, np.uint8))
    ink = pp.clean_page(raw & paper.astype(bool), char_h)
    columns = pp.find_columns(ink, char_h)

    for ln in lines:
        cx = float(np.median(ln.baseline[:, 0]))
        ln.column = next((ci for ci, (a, b) in enumerate(columns) if a <= cx < b), -1)

    # 1) brackets per column
    found, pitch = [], {}
    for ci, (x0, x1) in enumerate(columns):
        col = [l for l in lines if l.column == ci]
        if len(col) < 2:
            continue
        ys = sorted(l.y_mid for l in col)
        pitch[ci] = float(np.median(np.diff(ys)))
        col_ink = np.zeros_like(ink)
        col_ink[:, x0:x1] = ink[:, x0:x1]
        found += [(ci, br) for br in pp.find_brackets(col_ink, x0, x1, char_h, pitch[ci])]

    # 2) cut lines that Kraken ran through a bracket
    lines[:] = split_across_brackets(lines, found, pitch, char_h)

    # 3) members and label lines
    groups: list[Group] = []
    for ci, br in found:
        col = [l for l in lines if l.column == ci]
        p = pitch[ci]
        margin = float(np.percentile([l.x_start for l in col], 10))
        g = Group(f"g{len(groups) + 1:02d}", br, ci)
        # baselines sit at the foot of the letters, so the span starts at the bracket top
        lo, hi = br.y1, br.y2 + 0.25 * p
        m = (1 - core) / 2
        clo, chi = br.y1 + m * br.h, br.y2 - m * br.h
        max_start = margin + align_tol * (br.x1 - margin)
        for l in col:
            if lo <= l.y_mid <= hi and l.x_start <= max_start and l.x_end <= br.x2 + char_h:
                g.lines.append(l.id)
            elif clo <= l.y_mid <= chi and br.x1 + 0.5 * br.w <= l.x_start <= br.x2 + label_reach * char_h:
                g.label_lines.append(l.id)
        groups.append(g)
    return groups, {"char_h": char_h, "columns": columns}


# ---------------------------------------------------------------- 3. outputs
def draw(rgb: np.ndarray, lines: list[KLine], groups: list[Group], char_h: float,
         alpha: float = 0.22) -> np.ndarray:
    grouped = {i for g in groups for i in g.lines}
    labels = {i for g in groups for i in g.label_lines}
    fill = rgb.copy()
    for l in lines:
        colour = (150, 215, 170) if l.id in labels else (150, 175, 225)
        cv2.fillPoly(fill, [l.polygon.astype(np.int32)], colour)
    vis = cv2.addWeighted(fill, alpha, rgb, 1 - alpha, 0)
    t = max(1, int(round(char_h / 14)))
    for l in lines:
        if l.id in labels:
            colour, w = (0, 130, 60), t + 1
        elif l.id in grouped:
            colour, w = (40, 70, 170), t + 1
        else:
            colour, w = (95, 120, 185), t
        cv2.polylines(vis, [l.polygon.astype(np.int32)], True, colour, w, cv2.LINE_AA)
        cv2.polylines(vis, [l.baseline.astype(np.int32)], False, (220, 120, 40), t, cv2.LINE_AA)
    for g in groups:
        b = g.bracket
        cv2.rectangle(vis, (b.x1, b.y1), (b.x2, b.y2), (200, 40, 40), t)
    return vis


def to_json(stem: str, lines: list[KLine], groups: list[Group], meta: dict) -> dict:
    return {
        "image": stem,
        "columns": [[int(a), int(b)] for a, b in meta["columns"]],
        "lines": [{"id": l.id, "column": l.column,
                   "baseline": l.baseline.round().astype(int).tolist(),
                   "polygon": l.polygon.round().astype(int).tolist()} for l in lines],
        "bracket_groups": [{"id": g.id, "column": g.column,
                            "bracket": [g.bracket.x1, g.bracket.y1, g.bracket.x2, g.bracket.y2],
                            "lines": g.lines, "label_lines": g.label_lines} for g in groups],
    }


def run(image: str | Path = DEFAULT_IMAGE, out_dir: str | Path = DEFAULT_OUT,
        model: str | None = None, xml: bool = False, tight: bool = True,
        pagexml: str | Path | None = None, verbose: bool = True) -> dict:
    """pagexml: use lines from an eScriptorium/Kraken PAGE-XML export instead of running Kraken.
    Their outlines are kept exactly as exported (no extension or tightening)."""
    path = pp.resolve_image(image)
    img = Image.open(path).convert("RGB")
    rgb = np.array(img)
    if pagexml:
        lines = load_pagexml(pagexml)
        tight, skip_extend = False, True
    else:
        lines = kraken_segment(img, model)
        skip_extend = False
    raw = pp.ink_mask(rgb)
    char_h = pp.estimate_char_height(raw)
    paper = cv2.erode(pp.page_mask(rgb).astype(np.uint8), np.ones((int(char_h),) * 2, np.uint8))
    n_ext = 0 if skip_extend else extend_baselines_left(
        lines, pp.clean_page(raw & paper.astype(bool), char_h), char_h)
    if n_ext:
        recompute_boundaries(img, lines)
    groups, meta = bracket_groups(rgb, lines)
    if tight:
        tighten_boundaries(lines, rgb.shape[:2], meta["char_h"],
                           ink=pp.clean_page(raw & paper.astype(bool), char_h))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    Image.fromarray(draw(rgb, lines, groups, meta["char_h"])).save(out_dir / f"{stem}_kraken.png")
    (out_dir / f"{stem}_kraken.json").write_text(json.dumps(to_json(stem, lines, groups, meta), indent=1))
    if xml:
        write_pagexml(img, path.name, out_dir / f"{stem}.xml")
    if verbose:
        print(f"[{stem}] Kraken lines: {len(lines)} | extended left: {n_ext} | bracket groups: {len(groups)}")
        for g in groups:
            print(f"  {g.id} (column {g.column}): {len(g.lines)} lines, label lines {g.label_lines}")
        print(f"[{stem}] wrote {out_dir}")
    return {"lines": lines, "groups": groups, **meta}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Kraken line polygons + bracket groups.")
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", help="custom Kraken segmentation model (.mlmodel)")
    ap.add_argument("--xml", action="store_true", help="also write PAGE-XML for eScriptorium")
    ap.add_argument("--loose", action="store_true", help="keep Kraken's original (overlapping) outlines")
    ap.add_argument("--pagexml", help="use lines from an eScriptorium PAGE-XML export instead of Kraken")
    a = ap.parse_args()
    run(a.image, a.out, a.model, a.xml, tight=not a.loose, pagexml=a.pagexml)