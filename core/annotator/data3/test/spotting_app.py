"""
Word-spotting annotator (spotting_app.py)

Similar words are found by YOUR word spotting (batch_processing.run_spotting):
segmentation, handwriting normalisation, whitened HOG + column-profile DTW,
query-adaptive z-scores, phrase context, Louvain clusters, prune and merge.
This app adds, on top of its clusters:

  - the annotator interface: go through the clusters one by one, move / remove
    words, new clusters, merge / remove / restore clusters, concepts and
    signifiers, export (CSV or JSON)
  - a fit check like app.py's: mean similarity of a word to the rest of its
    cluster vs. to every other cluster (dense, not one edge at a time); the
    spotting's accepted matches still overrule it ("fits #N better",
    "fits no cluster")
  - "Transcribe with LLM" per cluster (DeepSeek), shown under every word
  - Kraken lines with bracket groups and labels, in their own tab (optional)

Put this file next to batch_processing.py (core/comparing_all_similar_words/)
and run from the project root:

    pip install streamlit openai
    streamlit run core/comparing_all_similar_words/spotting_app.py

Needs transcribe_clusters.py next to it (DeepSeek helpers); for the Kraken
tab also kraken_lines.py and page_polygons.py.

Everything for a set of pages is kept in app_runs/<pages>/:
  spotting/      the outputs of run_spotting (clusters.csv, clusters_<page>.jpg,
                 match_cache.pkl, ...) plus edges.json (every accepted match),
                 cands_<tag>.npz (features and boxes of every candidate, for
                 the fit check) and the segmented page
                 images the boxes refer to
  spotting_annotation.json  the annotation (clusters, moves, concepts, readings,
                 history) - its own file, separate from app.py's annotation.json
"""
from __future__ import annotations

import csv
import inspect
import io
import json
import os
import sys
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent


def find_project_root() -> Path | None:
    """The SETS project root: the nearest folder above this file that contains
    core/comparing_all_similar_words/batch_processing.py (override with the
    environment variable SETS_ROOT). batch_processing imports core.*, so the
    root must be on sys.path wherever this app is placed."""
    env = os.environ.get("SETS_ROOT")
    for cand in ([Path(env)] if env else []) + [HERE, *HERE.parents]:
        if (cand / "core" / "comparing_all_similar_words" / "batch_processing.py").is_file():
            return cand
    return None


ROOT = find_project_root()
for p in (HERE, ROOT):                                 # sibling modules, project root
    if p is not None and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if ROOT is not None:                                   # inside the SETS project
    from core.comparing_all_similar_words import batch_processing as bp
else:                                                  # batch_processing.py next to this file
    try:
        import batch_processing as bp
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "batch_processing not found: no folder above this app contains "
            "core/comparing_all_similar_words/batch_processing.py, and there is no "
            "batch_processing.py next to it. Set SETS_ROOT to the project folder "
            "(the one that contains 'core').") from e


SKIP_DIRS = {"venv", ".venv", "env", ".git", "__pycache__", "node_modules", "site-packages",
             "app_runs", "app_runs_spotting", ".idea"}


def find_module_file(filename: str) -> Path | None:
    """Where a helper module sits, searched in this order:
      1. this app's folder, then each folder above it up to the project root
      2. folders below this app's folder (e.g. the app in data3/test/ and the
         helper in data3/test/new/)
      3. anywhere else in the project (virtual environments and run folders skipped)"""
    top = ROOT if ROOT is not None else HERE
    for d in [HERE, *HERE.parents]:
        if (d / filename).is_file():
            return d
        if d == top:
            break

    def below(base: Path):
        hits = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(n for n in dirnames if n not in SKIP_DIRS and not n.startswith("."))
            if filename in filenames:
                hits.append(Path(dirpath))
        return sorted(hits, key=lambda p: (len(p.parts), str(p)))   # nearest first

    hits = below(HERE) or (below(top) if top != HERE else [])
    return hits[0] if hits else None


def add_module_dir(filename: str) -> Path | None:
    """Make a helper module importable wherever it sits in the project."""
    d = find_module_file(filename)
    if d is not None and str(d) not in sys.path:
        sys.path.append(str(d))
    return d


if add_module_dir("transcribe_clusters.py") is None:
    raise ModuleNotFoundError(
        f"transcribe_clusters.py was not found: not in {HERE}, not in a folder above it, "
        f"not in a folder below it, and not anywhere in the project "
        f"({ROOT or HERE}). Put transcribe_clusters.py next to spotting_app.py.")
import transcribe_clusters as tc                       # noqa: E402

APP_DIR = HERE
RUNS = APP_DIR / "app_runs_spotting"     # own folder: never mixes with app.py's annotations
OPEN = ("pending", "labelled")

FULL = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
        else {"use_container_width": True})

VERDICT = {"keep": ("✅", "fits"), "single": ("➖", "only word"),
           "move": ("↪️", "fits #{n} better"), "remove": ("❌", "fits no cluster")}
EXISTING, NEW = "Existing concept", "New concept"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================== #
# Pages and the spotting run
# =========================================================================== #
def run_dir(stem: str) -> Path:
    d = RUNS / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(upload) -> Path:
    """Each uploaded page is stored once, under app_runs/pages/ (original format)."""
    path = run_dir("pages") / Path(upload.name).name
    if not path.exists():
        path.write_bytes(upload.getvalue())
    return path


def doc_name(stems: list[str]) -> str:
    """Folder name for a set of pages (one page: its name)."""
    import hashlib
    stems = sorted(stems)
    if len(stems) == 1:
        return stems[0]
    joined = "+".join(stems)
    return joined if len(joined) <= 60 else \
        f"{stems[0]}+{len(stems) - 1}more_{hashlib.md5(joined.encode()).hexdigest()[:6]}"


SPOT_FIELDS = {f.name for f in fields(bp.CrossParams)}


@st.cache_data(show_spinner=False)
def step_spotting(name: str, images: tuple[str, ...], params: tuple, within: bool, cross: bool,
                  n_jobs: int, mtimes: tuple) -> dict:
    """Run your word spotting on the pages and keep what the app needs:
    clusters.csv (written by run_spotting), the segmented page images the
    boxes refer to, and every accepted match (edges.json, for the fit check).

    run_spotting caches its matches in match_cache.pkl, so changing only the
    clustering settings re-clusters in seconds."""
    out = run_dir(name) / "spotting"
    cp = bp.CrossParams(**dict(params))
    res = bp.run_spotting(list(images), out, cp, within=within, cross=cross,
                          n_jobs=n_jobs, use_cache=True, verbose=False)
    pages = {}
    for P in res["pages"]:                         # boxes refer to the segmented image
        path = out / f"seg_{P.name}.png"
        cv2.imwrite(str(path), P.img)
        pages[P.name] = str(path)
    edges = {}
    for P, Q, pairs in res["pairs"]:
        for m in pairs:
            a = f"{P.name}:{P.cands[m['i']].parent}"
            b = f"{Q.name}:{Q.cands[m['j']].parent}"
            if a != b:
                key = "|".join(sorted((a, b)))
                edges[key] = max(edges.get(key, -np.inf), float(m["z"]))
    (out / "edges.json").write_text(json.dumps(edges))
    # clusters.csv / edges.json are overwritten by every run in this folder (the app with
    # other settings, check_spotting_app.py). st.cache_data can skip run_spotting and
    # return these paths later, so keep a copy per settings and point to that instead.
    import hashlib, shutil
    tag = hashlib.md5(repr((params, within, cross)).encode()).hexdigest()[:8]
    shutil.copyfile(out / "clusters.csv", out / f"clusters_{tag}.csv")
    shutil.copyfile(out / "edges.json", out / f"edges_{tag}.json")
    save_candidates(out / f"cands_{tag}.npz", res["pages"])
    return {"clusters_csv": str(out / f"clusters_{tag}.csv"),
            "edges": str(out / f"edges_{tag}.json"),
            "cands": str(out / f"cands_{tag}.npz"),
            "pages": pages,
            "overlays": {n: str(out / f"clusters_{n}.jpg") for n in pages},
            "summary": read_summary(out / "summary.csv"),
            "params": dict(params)}


def read_summary(path: Path) -> list[dict]:
    """summary.csv of run_spotting: one row per comparison (a page with
    itself = within, two pages = cross) with its number of accepted matches."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [dict(page_a=r["page_a"], page_b=r["page_b"], type=r["type"],
                         n_matches=int(r["n_matches"])) for r in csv.DictReader(f)]
    except (OSError, KeyError, ValueError):
        return []


def explain_no_clusters(S: dict, stems: list[str], within: bool, cross: bool) -> None:
    """Why clusters.csv is empty: which pages the spotting used, and how many
    matches each comparison accepted - with the setting to change."""
    used = set(S["pages"])
    skipped = [s for s in stems if s not in used]
    rows = S.get("summary") or []
    p = S.get("params") or {}
    total = sum(r["n_matches"] for r in rows)
    lines = ["**No word clusters.** What the word spotting found:"]
    if skipped:
        lines.append(f"- page(s) **skipped** by batch_processing (segmentation failed or no "
                     f"word candidates): {', '.join(skipped)} - run batch_processing on them "
                     f"in a terminal to see why")
    if not within and not cross:
        lines.append("- both *Similar words inside each page* and *between pages* are off")
    elif not within:
        lines.append("- *Similar words inside each page* is off")
    elif not cross and len(stems) > 1:
        lines.append("- *Similar words between pages* is off")
    if rows:
        lines.append("")
        lines.append("| comparison | pages | accepted matches |")
        lines.append("|---|---|---|")
        for r in rows:
            pages_ = r["page_a"] if r["type"] == "within" else f"{r['page_a']} ↔ {r['page_b']}"
            lines.append(f"| {r['type']} | {pages_} | {r['n_matches']} |")
        lines.append("")
    if rows and total == 0:
        lines.append(f"No comparison accepted a match (z threshold {p.get('z_threshold')}, "
                     f"score floor {p.get('score_floor')}, margin {p.get('margin')}). With "
                     f"several pages the HOG statistics are fitted on all pages together, so "
                     f"pages that differ a lot (resolution, lighting) score lower than alone. "
                     f"In the sidebar (*Word spotting*): lower the **z threshold** (e.g. 2.5) "
                     f"and the **score floor**, or raise the **margin**.")
    elif total > 0:
        cmz, zt = p.get("cluster_min_z"), p.get("z_threshold")
        if cmz is not None and zt is not None and cmz > zt:
            lines.append(f"{total} matches were accepted, but none reached cluster min z "
                         f"{cmz}: tick **Use every accepted pair for clusters** or lower "
                         f"*Cluster min z*.")
        else:
            lines.append(f"{total} matches were accepted, but the clustering kept none: "
                         f"lower *Prune: min link fraction* or the *Louvain resolution*.")
    st.warning("\n".join(lines))


def read_clusters(path: str) -> dict:
    """clusters.csv -> the word format the annotation uses:
    {"words": [{id, page, token, line, box, quad}], "clusters": {number: [ids]}}.
    A word is a token of a page ("<page>:<token>"); its box is the matched span."""
    words, clusters = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wid = f"{r['page']}:{r['token']}"
            x0, y0, x1, y1 = (int(float(r[k])) for k in ("x0", "y0", "x1", "y1"))
            words[wid] = dict(id=wid, page=r["page"], token=r["token"], line=r["token"],
                              candidate=r.get("candidate"), box=[x0, y0, x1, y1],
                              quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                              best_z=float(r["best_z"]) if r.get("best_z") else None,
                              origin="spotting")
            clusters.setdefault(str(int(r["cluster"])), []).append(wid)
    return {"words": list(words.values()), "clusters": clusters}


# =========================================================================== #
# Token features (the spotting's own whitened HOG) for the fit check
# =========================================================================== #
def save_candidates(path: Path, pages: list) -> None:
    """Features and boxes of EVERY candidate of every page (matched or not).
    The whitened HOG features are the ones the spotting compared, so the fit
    check works in the spotting's own feature space."""
    rows = [(P, c) for P in pages for c in P.cands if c.feat is not None]
    if not rows:
        return
    np.savez_compressed(
        path,
        page=np.array([P.name for P, _ in rows]),
        tid=np.array([c.tid for _, c in rows]),
        parent=np.array([c.parent for _, c in rows]),
        full=np.array([c.is_full for _, c in rows]),
        box=np.array([[c.box.x0, c.box.y0, c.box.x1, c.box.y1] for _, c in rows], np.int32),
        width=np.array([c.box.width / max(P.unit, 1.0) for P, c in rows], np.float32),
        feat=np.stack([np.asarray(c.feat, np.float32).ravel() for _, c in rows]))


def token_table(path: str, chosen: dict[str, str]) -> dict:
    """One row per segmented token, from cands_<tag>.npz:
    the candidate the spotting reported for clustered tokens (`chosen`:
    word id -> candidate id), otherwise the whole token, otherwise its widest
    candidate. Features are L2-normalised, so a dot product is a cosine."""
    C = np.load(path)
    best: dict[str, tuple] = {}
    for k in range(len(C["tid"])):
        wid = f"{C['page'][k]}:{C['parent'][k]}"
        rank = (chosen.get(wid) == str(C["tid"][k]), bool(C["full"][k]), float(C["width"][k]))
        if wid not in best or rank > best[wid][0]:
            best[wid] = (rank, k)
    ids = list(best)
    idx = np.array([best[w][1] for w in ids], int)
    F = C["feat"][idx].astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    return dict(ids=ids, pos={w: i for i, w in enumerate(ids)}, F=F,
                box=C["box"][idx], width=C["width"][idx],
                page=[str(p) for p in C["page"][idx]], token=[str(t) for t in C["parent"][idx]],
                tid=[str(t) for t in C["tid"][idx]])


WIDTH_RATIO = 1.3   # fit check: "fits #N better" only for clusters of comparable width


def similarity_stats(F: np.ndarray, n_pairs: int = 200_000, seed: int = 0) -> tuple[float, float]:
    """Mean and SD of the cosine between random token pairs: the background
    every similarity is turned into a z-score against (scale-free, as in
    batch_processing.merge_clusters)."""
    n = len(F)
    if n < 2:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = a != b
    s = np.einsum("ij,ij->i", F[a[keep]], F[b[keep]])
    return float(s.mean()), float(s.std() + 1e-8)


def width_ok(wa: np.ndarray, wb: np.ndarray, ratio: float) -> np.ndarray:
    """True where two box widths (in line pitches) differ by at most `ratio`."""
    wa, wb = np.maximum(wa, 1e-3), np.maximum(wb, 1e-3)
    return np.maximum(wa / wb, wb / wa) <= ratio


def cluster_profile(T: dict, groups: dict[str, list[str]]) -> tuple[list[str], np.ndarray,
                                                                    np.ndarray, np.ndarray]:
    """Per cluster: summed member features, size and median width (members
    missing from the token table are ignored)."""
    cids, sums, sizes, widths = [], [], [], []
    for cid, members in groups.items():
        rows = [T["pos"][w] for w in members if w in T["pos"]]
        if not rows:
            continue
        cids.append(cid)
        sums.append(T["F"][rows].sum(0))
        sizes.append(len(rows))
        widths.append(float(np.median(T["width"][rows])))
    d = T["F"].shape[1]
    return (cids, np.array(sums, np.float32).reshape(-1, d), np.array(sizes, float),
            np.array(widths, float))


@st.cache_data(show_spinner=False)
def _cached_table(path: str, chosen: tuple, mtime: float) -> dict:
    return token_table(path, dict(chosen))


def table_for(path: str, words: dict) -> dict | None:
    """Token table for the fit check (None if cands_<tag>.npz is missing)."""
    if not Path(path).is_file():
        return None
    chosen = tuple(sorted((w, str(v["candidate"])) for w, v in words.items()
                          if v.get("candidate")))
    return _cached_table(path, chosen, Path(path).stat().st_mtime)


@st.cache_data(show_spinner=False)
def page_rgb(img_path: str, mtime: float) -> np.ndarray:
    return np.array(Image.open(img_path).convert("RGB"))


@st.cache_data(show_spinner=False)
def step_lines(img_path: str, mtime: float) -> dict:
    """Kraken lines + bracket groups of one page (kraken_lines.py), on request."""
    for f in ("kraken_lines.py", "page_polygons.py"):
        add_module_dir(f)
    import kraken_lines as kl
    d = run_dir("pages") / "lines"
    try:                               # tight, non-overlapping outlines (as app.py)
        kl.run(img_path, d, tight=True, verbose=False)
    except TypeError:                  # older kraken_lines.run without `tight`
        kl.run(img_path, d, verbose=False)
    stem = Path(img_path).stem
    return {"json": str(d / f"{stem}_kraken.json"), "overlay": str(d / f"{stem}_kraken.png")}


# =========================================================================== #
# Kraken lines -> words: each line cut into pieces at the white space
# =========================================================================== #
SLANTS = tuple(range(-10, 46, 5))          # candidate slants (degrees, right-leaning > 0)


def baseline_at(line: dict, xs: np.ndarray) -> np.ndarray | None:
    """y of the line's Kraken baseline at every x (None if it has none)."""
    bl = line.get("baseline")
    if not bl or len(bl) < 2:
        return None
    b = np.asarray(bl, float).reshape(-1, 2)
    b = b[np.argsort(b[:, 0])]
    return np.interp(xs, b[:, 0], b[:, 1])


def line_ink(gray: np.ndarray, poly: np.ndarray, min_speck: float, pad: float = 0.5):
    """Whole strokes of one line. Ink is looked for in the line polygon
    widened by `pad` x its height, but only strokes (connected components)
    that have ink INSIDE the polygon are kept: a capital's loop or a tail
    that leaves the tight Kraken outline stays whole, a neighbouring line's
    word that never touches it is ignored.

    Returns ys, xs (page coordinates), the stroke id and inside-flag of
    every pixel, the strokes that run out of the widened area (they
    continue into another line), the polygon mask and its bounding box."""
    H, W = gray.shape
    px0, py0 = np.floor(poly.min(0)).astype(int)
    px1, py1 = np.ceil(poly.max(0)).astype(int)
    p = int(pad * (py1 - py0))
    x0, y0, x1, y1 = max(px0 - p, 0), max(py0 - p, 0), min(px1 + p, W), min(py1 + p, H)
    none = np.empty(0, int)
    empty = (none, none, none, np.empty(0, bool), set(), None, (x0, y0, x1, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return empty
    crop = gray[y0:y1, x0:x1]
    inside = np.zeros(crop.shape, np.uint8)
    cv2.fillPoly(inside, [np.round(poly - (x0, y0)).astype(np.int32)], 1)
    vals = crop[inside > 0]
    if vals.size < 10:
        return empty
    widened = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2 * p + 1, 2 * p + 1)))
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = ((crop < thr) & (widened > 0)).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h = py1 - py0
    hits = np.bincount(lab[(inside > 0) & (ink > 0)], minlength=len(stats))
    keep = (hits > 0) & (stats[:, cv2.CC_STAT_AREA] >= min_speck * h * h)
    keep[0] = False
    ink = keep[lab].astype(np.uint8)
    rim = widened - cv2.erode(widened, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT,
                              borderValue=0)
    edge = np.zeros_like(rim)                             # page edge counts as rim too
    edge[[0, -1], :] = 1
    edge[:, [0, -1]] = 1
    runs_out = set(np.unique(lab[((rim > 0) | (edge > 0)) & (ink > 0)]).tolist()) - {0}
    ys, xs = np.nonzero(ink)
    return (ys + y0, xs + x0, lab[ys, xs], inside[ys, xs] > 0, runs_out, inside,
            (x0, y0, x1, y1))


def core_band(ys: np.ndarray, bl: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    """The body of the writing (x-height band, just above the baseline):
    returns each pixel's height above the baseline, a mask of the pixels in
    the band and the x-height. Gaps are looked for only here - ascenders and
    descenders, including those of the neighbouring lines that reach into
    this polygon, would otherwise bridge the space between words.
    Without a baseline, the bottom of the densest rows stands in for it."""
    if bl is None:
        rows = np.bincount(ys - ys.min())
        core = np.flatnonzero(rows >= 0.5 * rows.max())
        bl = np.full(len(ys), ys.min() + core[-1], float)
    dy = bl - ys                                        # > 0 above the baseline
    near = (dy > -0.5 * np.ptp(ys)) & (dy < np.ptp(ys))
    d = np.round(dy[near]).astype(int)
    if d.size == 0:
        return dy, np.ones(len(ys), bool), 4.0
    hist = np.bincount(d - d.min())
    core = np.flatnonzero(hist >= 0.5 * hist.max()) + d.min()
    lo, hi = float(core.min()), float(core.max())
    xh = max(hi - lo + 1, 4.0)
    return dy, (dy >= lo - 0.15 * xh) & (dy <= hi + 0.15 * xh), xh


def core_band_all(ys: np.ndarray, xs: np.ndarray, line: dict,
                  inside: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """core_band measured on the pixels INSIDE the outline (strokes reaching
    out of it do not shift the band), then applied to every pixel."""
    bl = baseline_at(line, xs)
    dy_in, core_in, xh = core_band(ys[inside], None if bl is None else bl[inside])
    lo, hi = dy_in[core_in].min(), dy_in[core_in].max()
    if bl is None:                                # same stand-in baseline as core_band
        rows = np.bincount(ys[inside] - ys[inside].min())
        c = np.flatnonzero(rows >= 0.5 * rows.max())
        bl = np.full(len(ys), ys[inside].min() + c[-1], float)
    dy = bl - ys
    return dy, (dy >= lo) & (dy <= hi), xh


def sheared(xs: np.ndarray, ys: np.ndarray, yc: float, slant: float) -> np.ndarray:
    """x after removing the slant (right-leaning strokes stand upright)."""
    return xs + (ys - yc) * np.tan(np.radians(slant))


def find_segments(xp: np.ndarray, min_gap: float) -> list[tuple[float, float]]:
    """[start, end) runs of ink along x, separated by empty runs >= min_gap."""
    base = int(np.floor(xp.min()))
    filled = np.bincount(np.round(xp).astype(int) - base) > 0
    segs, start, gap = [], None, 0
    for x, f in enumerate(filled):
        if f:
            if start is None:
                start = x
            elif gap >= min_gap:
                segs.append((start + base, x - gap + base))
                start = x
            gap = 0
        else:
            gap += 1
    if start is not None:
        segs.append((start + base, len(filled) - gap + base))
    return segs


def best_slant(xs: np.ndarray, ys: np.ndarray, yc: float) -> float:
    """The slant at which the core band shows the most empty columns: the
    strokes stand upright there, so the gaps between words open widest."""
    def empty(a):
        xp = np.round(sheared(xs, ys, yc, a)).astype(int)
        return int((np.bincount(xp - xp.min()) == 0).sum())
    return float(max(SLANTS, key=empty))


def merge_narrow(segs: list[tuple[float, float]], min_w: float,
                 max_gap: float) -> list[tuple[float, float]]:
    """Segments narrower than min_w join the neighbour across the smaller gap
    (an initial, an 'S.', a flourish split off), if that gap is < max_gap;
    a narrow piece far from everything stays on its own."""
    segs = list(segs)
    while len(segs) > 1:
        for k, (a, b) in enumerate(segs):
            if b - a >= min_w:
                continue
            left = a - segs[k - 1][1] if k > 0 else np.inf
            right = segs[k + 1][0] - b if k + 1 < len(segs) else np.inf
            if min(left, right) >= max_gap:
                continue
            lo, hi = (k - 1, k) if left <= right else (k, k + 1)
            segs[lo:hi + 1] = [(segs[lo][0], segs[hi][1])]
            break
        else:
            break
    return segs


def piece_polygon(inside: np.ndarray, bbox: tuple, yc: float, slant: float,
                  left: float, right: float, wys: np.ndarray, wxs: np.ndarray) -> list[list[int]]:
    """The word's region as a polygon (page coordinates): the slice of the
    line outline between two slanted cuts, grown by the word's own strokes
    (loops and tails that leave the outline or swing past a cut)."""
    x0, y0, x1, y1 = bbox
    Y, X = np.mgrid[y0:y1, x0:x1]
    xp = sheared(X, Y, yc, slant)
    mask = inside & ((xp >= left) & (xp < right)).astype(np.uint8)
    strokes = np.zeros_like(mask)
    strokes[wys - y0, wxs - x0] = 1
    mask |= cv2.dilate(strokes, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
    return (c + (x0, y0)).astype(int).tolist()


def assign_strokes(comp: np.ndarray, xp: np.ndarray, core: np.ndarray,
                   cuts: list[float]) -> np.ndarray:
    """Word index of every pixel. A stroke goes WHOLE to the word holding
    most of its core-band ink (or, with none, its middle), so a capital's
    loop or a tail crossing a cut stays with its word. A stroke whose core
    ink is shared between words (letters joined across a gap) is split at
    the cuts instead."""
    edges = np.asarray(cuts[1:-1], float)
    by_cut = np.searchsorted(edges, xp, side="right")     # word by position alone
    word = by_cut.copy()
    for c in np.unique(comp):
        px = comp == c
        k = by_cut[px & core]
        if k.size == 0:
            word[px] = np.searchsorted(edges, np.median(xp[px]), side="right")
            continue
        counts = np.bincount(k)
        top = np.sort(counts)[::-1]
        if len(top) == 1 or top[1] < 0.25 * top[0]:
            word[px] = int(counts.argmax())
    return word


def bridged(segs: list[tuple[float, float]], xc: np.ndarray, cc: np.ndarray,
            min_px: float) -> list[bool]:
    """For every gap between consecutive segments: is it crossed by one
    continuous stroke? True when a single connected component has at least
    `min_px` core-band pixels on BOTH sides - letters joined above or below
    the core band, or a capital whose loop runs over the next letter. Such
    a gap is inside a word, however wide it is in the core band."""
    out = []
    for (a0, b0), (a1, b1) in zip(segs, segs[1:]):
        left = np.bincount(cc[(xc >= a0) & (xc <= b0)])
        right = np.bincount(cc[(xc >= a1) & (xc <= b1)])
        n = min(len(left), len(right))
        out.append(bool(n and np.any((left[:n] >= min_px) & (right[:n] >= min_px))))
    return out


def join_bridged(segs: list[tuple[float, float]], xc: np.ndarray, cc: np.ndarray,
                 min_px: float) -> list[tuple[float, float]]:
    """Segments separated by a bridged gap (see bridged) become one."""
    if len(segs) < 2:
        return segs
    out = [segs[0]]
    for seg, br in zip(segs[1:], bridged(segs, xc, cc, min_px)):
        out[-1:] = [(out[-1][0], seg[1])] if br else [out[-1], seg]
    return out


def word_gap_threshold(gaps: np.ndarray) -> float | None:
    """The gap (in x-heights) that separates the spaces BETWEEN words from
    the blanks between unjoined letters INSIDE words, from all the gaps of
    the page: Otsu's threshold on their logarithms (the two groups differ
    by a factor, not by a fixed amount). None if the page does not show two
    distinct groups (too few gaps, or the groups lie too close together)."""
    g = gaps[(gaps > 0.05) & (gaps < 6.0)]
    if len(g) < 15:
        return None
    lg = np.log(g)
    cand = np.unique(lg)[:-1]
    best, best_var = None, -1.0
    for c in cand:
        lo, hi = lg[lg <= c], lg[lg > c]
        var = len(lo) * len(hi) * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best, best_var = c, var
    if best is None:
        return None
    lo, hi = g[lg <= best], g[lg > best]
    if hi.mean() < 1.8 * lo.mean():               # not two groups: no automatic value
        return None
    t = float(np.exp((best + lg[lg > best].min()) / 2))   # middle of the empty stretch
    return float(np.clip(t, 0.25, 2.5))


BRIDGE = 0.2       # core-band ink (x x-height^2) a stroke needs on both sides of a gap


def prepare_line(gray: np.ndarray, line: dict, n: int, slant: float | None,
                 min_speck: float, keep_strokes: bool = True) -> dict | None:
    """Everything about one line that does not depend on where the cuts go:
    its own strokes (neighbouring lines' strokes removed), core band,
    x-height, slant, sheared x of every pixel, and the gaps (in x-heights)
    between the runs of core ink."""
    poly = np.asarray(line["polygon"], float).reshape(-1, 2)
    ys, xs, comp, inside, runs_out, mask, bbox = line_ink(gray, poly, min_speck)
    if len(xs) < 5 or inside.sum() < 5:
        return None
    dy, core, xh = core_band_all(ys, xs, line, inside)
    # strokes of the neighbouring lines: little ink in this line's core band
    # and either running on beyond the widened outline or mostly outside it
    n_all = np.bincount(comp)
    n_core = np.bincount(comp[core], minlength=len(n_all))
    n_in = np.bincount(comp[inside], minlength=len(n_all))
    foreign = (n_core < 0.5 * n_all) & (n_in < 0.5 * n_all)
    for c in runs_out:
        foreign[c] |= n_core[c] < 0.5 * n_all[c]
    keep = ~foreign[comp]
    ys, xs, comp, core = ys[keep], xs[keep], comp[keep], core[keep]
    if core.sum() < 5:
        return None
    yc = float(np.median(ys[core]))
    a = best_slant(xs[core], ys[core], yc) if slant is None else float(slant)
    xp = sheared(xs, ys, yc, a)
    fine = find_segments(xp[core], 2.0)
    if keep_strokes:              # gaps crossed by a stroke are not candidate word gaps
        fine = join_bridged(fine, xp[core], comp[core], BRIDGE * xh * xh)
    gaps = np.array([(fine[k + 1][0] - fine[k][1]) / xh for k in range(len(fine) - 1)])
    return dict(lid=str(line.get("id", n)), ys=ys, xs=xs, comp=comp, core=core, xh=xh,
                yc=yc, slant=a, xp=xp, mask=mask, bbox=bbox, gaps=gaps,
                keep_strokes=keep_strokes)


def cut_line(L: dict, gap: float, min_word: float, min_mass: float) -> list[dict]:
    """Cut one prepared line at every gap of at least `gap` x-heights and
    return its words (see kraken_words)."""
    ys, xs, comp, core, xh, xp = L["ys"], L["xs"], L["comp"], L["core"], L["xh"], L["xp"]
    min_gap = max(2.0, gap * xh)
    xc = xp[core]
    segs = find_segments(xc, min_gap)
    if L["keep_strokes"]:
        segs = join_bridged(segs, xc, comp[core], BRIDGE * xh * xh)
    segs = [(a0, b0) for a0, b0 in segs
            if ((xc >= a0) & (xc < b0 + 1)).sum() >= min_mass * xh * xh]
    if not segs:
        return []
    segs = merge_narrow(segs, min_word * xh, 2 * min_gap)
    cuts = [-np.inf] + [(segs[k][1] + segs[k + 1][0]) / 2 for k in range(len(segs) - 1)] \
        + [np.inf]
    word_of = assign_strokes(comp, xp, core, cuts)
    # a stroke with no core ink outside every word (a lone tail) belongs to none
    lone = ~np.isin(comp, np.unique(comp[core]))
    words = []
    for k in range(len(segs)):
        sel = (word_of == k) & ~(lone & ((xp < segs[0][0] - min_gap) |
                                         (xp > segs[-1][1] + min_gap)))
        if not sel.any():
            continue
        bx0, bx1 = int(xs[sel].min()), int(xs[sel].max()) + 1
        by0, by1 = int(ys[sel].min()), int(ys[sel].max()) + 1
        left = segs[k][0] - 1 if k == 0 else cuts[k]
        right = segs[k][1] + 1 if k == len(segs) - 1 else cuts[k + 1]
        words.append(dict(id=f"{L['lid']}_w{k + 1}", line=L["lid"],
                          polygon=piece_polygon(L["mask"], L["bbox"], L["yc"], L["slant"],
                                                left, right, ys[sel], xs[sel]),
                          box=[bx0, by0, bx1, by1],
                          quad=[[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]],
                          slant=L["slant"], x_height=round(xh, 1), gap=round(gap, 2)))
    return words


@st.cache_data(show_spinner=False)
def kraken_words(img_path: str, lines_json: str, gap: float | None, slant: float | None,
                 min_speck: float, min_word: float, min_mass: float,
                 mtime: float, fallback_gap: float = 0.5,
                 keep_strokes: bool = True) -> list[dict]:
    """Cut every Kraken line into words at the white space between them.

    1. whole strokes of the line: ink in the outline widened by half its
       height, kept if the stroke has ink inside the outline, dropped if it
       belongs to a neighbouring line (little ink in this line's core band
       and running on beyond the widened outline, or mostly outside it);
    2. the core band just above the baseline (x-height): only there are
       gaps looked for, so ascenders and descenders cannot fill them;
    3. slant: per line, the shear that opens the most empty columns (or a
       fixed one), so the cuts follow the slope of the writing;
    4. which gaps are spaces between words: gap=None measures every gap of
       the page and separates the two groups (blanks between unjoined
       letters inside a word, spaces between words - see
       word_gap_threshold), falling back to `fallback_gap` if the page shows
       no clear groups; a number is used as it is (x x-height);
       With keep_strokes, a gap crossed by one continuous stroke (letters
       joined outside the core band, a capital's loop) is never cut and
       does not count in that measurement;
    5. cuts in the middle of those gaps; pieces with less core ink than
       `min_mass` x x-height^2 (a lone tail) are dropped, pieces narrower
       than `min_word` x x-height join a close neighbour;
    6. each stroke goes whole to the word holding most of its core ink
       (capital loops, tails crossing a cut), and each word = the slice of
       the line outline between two cuts, grown by its own strokes.

    Returns per word: id (<line>_w<k>), line, polygon (slice + strokes),
    box (around the word's strokes), quad, slant, x_height, gap (the gap
    threshold used, x x-height)."""
    gray = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    lines = [L for n, line in enumerate(json.loads(Path(lines_json).read_text())["lines"], 1)
             if (L := prepare_line(gray, line, n, slant, min_speck, keep_strokes)) is not None]
    if gap is None:
        all_gaps = np.concatenate([L["gaps"] for L in lines]) if lines else np.empty(0)
        gap = word_gap_threshold(all_gaps) or fallback_gap
    return [w for L in lines for w in cut_line(L, gap, min_word, min_mass)]


def load_joins(path: Path) -> list[list[float]]:
    """Joins the annotator made by hand on one page: points (page px) between
    two words of a line that must not be cut there."""
    try:
        return json.loads(path.read_text())["joins"]
    except (OSError, ValueError, KeyError):
        return []


def save_joins(path: Path, joins: list[list[float]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"joins": joins}))
    os.replace(tmp, path)


def join_point(a: dict, b: dict) -> list[float]:
    """The point between two neighbouring words that identifies their cut
    (kept as a point, so the join survives changes to the cut settings)."""
    return [(a["box"][2] + b["box"][0]) / 2, (a["box"][1] + a["box"][3] + b["box"][1]
                                              + b["box"][3]) / 4]


def merge_words(a: dict, b: dict) -> dict:
    """Two neighbouring words of a line as one (polygons and boxes united)."""
    x0, y0 = min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1])
    x1, y1 = max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])
    polys = [np.array(w["polygon"]) for w in (a, b) if len(w.get("polygon") or []) >= 3]
    poly = a.get("polygon")
    if polys:
        pts = np.vstack(polys)
        ox, oy = pts.min(0) - 2
        m = np.zeros((int(pts[:, 1].max() - oy) + 3, int(pts[:, 0].max() - ox) + 3), np.uint8)
        for p in polys:
            cv2.fillPoly(m, [(p - (ox, oy)).astype(np.int32)], 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 1.5, True).reshape(-1, 2)
        poly = (c + (ox, oy)).astype(int).tolist()
    return dict(a, polygon=poly, box=[x0, y0, x1, y1],
                quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], joined=True)


def apply_joins(words: list[dict], joins: list[list[float]]) -> list[dict]:
    """Merge the neighbouring words (same line) at every saved join point.
    Each point undoes exactly ONE cut: the one between two words of a line
    whose height contains the point and whose boundary is nearest to it
    (no further than the narrower of the two words is wide)."""
    merge_after = set()                         # index i: merge words[i] and words[i + 1]
    for x, y in joins:
        best, best_d = None, np.inf
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a["line"] != b["line"]:
                continue
            if not min(a["box"][1], b["box"][1]) <= y <= max(a["box"][3], b["box"][3]):
                continue
            d = abs(x - (a["box"][2] + b["box"][0]) / 2)
            limit = min(a["box"][2] - a["box"][0], b["box"][2] - b["box"][0])
            if d <= limit and d < best_d:
                best, best_d = i, d
        if best is not None:
            merge_after.add(best)
    out: list[dict] = []
    for i, w in enumerate(words):
        if i - 1 in merge_after and out:
            out[-1] = merge_words(out[-1], w)
        else:
            out.append(w)
    return out


def join_words(path: str, point: list[float]) -> None:
    save_joins(Path(path), load_joins(Path(path)) + [point])
    flash("ok", "words joined")


def undo_joins(path: str, line_words: list[dict]) -> None:
    """Remove the hand-made joins inside the given words (one line)."""
    keep = [p for p in load_joins(Path(path))
            if not any(w["box"][0] <= p[0] <= w["box"][2] and w["box"][1] <= p[1] <= w["box"][3]
                       for w in line_words)]
    save_joins(Path(path), keep)
    flash("ok", "joins of this line undone")


WORD_COLOURS = [(230, 60, 60), (30, 120, 230), (30, 170, 70)]


def draw_kraken_words(rgb: np.ndarray, lines_json: str, words: list[dict], numbers: bool,
                      height: int = 2000) -> np.ndarray:
    """The page with every Kraken line shown as its word pieces: each slice
    of the line tinted and outlined, colours alternating along the line,
    the baseline in orange; optionally numbered by position in the line."""
    s = height / rgb.shape[0]
    canvas = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    tint = canvas.copy()
    for w in words:
        if len(w["polygon"]) >= 3:
            k = int(w["id"].rsplit("_w", 1)[1])
            cv2.fillPoly(tint, [np.round(np.array(w["polygon"]) * s).astype(np.int32)],
                         WORD_COLOURS[k % 3])
    canvas = cv2.addWeighted(tint, 0.22, canvas, 0.78, 0)
    for line in json.loads(Path(lines_json).read_text())["lines"]:
        if line.get("baseline"):
            bl = np.round(np.asarray(line["baseline"], float).reshape(-1, 2) * s).astype(np.int32)
            cv2.polylines(canvas, [bl], False, (240, 140, 30), 1)
    for w in words:
        if len(w["polygon"]) < 3:
            continue
        k = int(w["id"].rsplit("_w", 1)[1])
        pts = np.round(np.array(w["polygon"]) * s).astype(np.int32)
        cv2.polylines(canvas, [pts], True, WORD_COLOURS[k % 3], 2)
        if numbers:
            x, y = pts[:, 0].min(), pts[:, 1].min()
            cv2.putText(canvas, str(k), (int(x) + 2, max(int(y) + 12, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WORD_COLOURS[k % 3], 1)
    return canvas


def word_crop(rgb: np.ndarray, w: dict) -> np.ndarray:
    """The word's slice of the line, white outside the slice."""
    pts = np.array(w["polygon"]) if len(w["polygon"]) >= 3 else np.array(w["quad"])
    x0, y0 = np.maximum(pts.min(0), 0)
    x1, y1 = pts.max(0) + 1
    crop = rgb[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [(pts - (x0, y0)).astype(np.int32)], 1)
    crop[mask == 0] = 255
    return crop


def kraken_word_view(path: Path, K: dict, key: str, ctx: dict | None = None) -> None:
    """Settings, page image and line-by-line gallery of the Kraken words;
    the words are also saved as <page>_kraken_words.json next to the lines.
    With ctx (the annotation), the page is clickable: words without a cluster
    are red and can be added to one."""
    c1, c2, c3, c4, c5 = st.columns(5)
    auto_gap = c1.toggle("Word gap: automatic (from the page)", True, key=f"kw_ag_{key}",
                         help="measures every gap of the page and separates the spaces "
                              "between words from the blanks inside words")
    gap = c1.slider("Min gap between words (× x-height)", 0.1, 2.5, 0.5, 0.05,
                    key=f"kw_gap_{key}", disabled=auto_gap,
                    help="used when the automatic value is off (or the page shows no "
                         "clear groups); raise it if words are cut in two")
    auto = c2.toggle("Slant: automatic per line", True, key=f"kw_auto_{key}")
    slant = None if auto else c2.slider("Slant (degrees)", -10, 45, 20, 1, key=f"kw_sl_{key}")
    min_speck = c3.slider("Ignore specks below (× height²)", 0.0, 0.05, 0.004, 0.001,
                          format="%.3f", key=f"kw_sp_{key}")
    min_word = c4.slider("Min word width (× x-height)", 0.0, 3.0, 0.8, 0.1,
                         key=f"kw_mw_{key}", help="narrower pieces join a close neighbour")
    keep_strokes = c1.toggle("Never cut through a connected stroke", True,
                             key=f"kw_ks_{key}",
                             help="a gap crossed by one continuous pen stroke (joined "
                                  "letters, a capital's loop) is inside a word")
    min_mass = c5.slider("Min ink of a word (× x-height²)", 0.0, 1.5, 0.3, 0.05,
                         key=f"kw_mm_{key}", help="lighter pieces (a lone tail or flourish "
                                                  "crossing the line) are not words")
    words = kraken_words(str(path), K["json"], None if auto_gap else gap, slant, min_speck,
                         min_word, min_mass, path.stat().st_mtime, fallback_gap=gap,
                         keep_strokes=keep_strokes)
    joins_path = Path(K["json"]).with_name(f"{path.stem}_kraken_joins.json")
    joins = load_joins(joins_path)
    words = apply_joins(words, joins)
    if words:
        st.caption(f"Words are cut at gaps of at least **{words[0]['gap']:.2f} × x-height**"
                   + (" (measured on this page)" if auto_gap and words[0]["gap"] != gap
                      else ""))
    out = Path(K["json"]).with_name(f"{path.stem}_kraken_words.json")
    out.write_text(json.dumps({"image": str(path), "words": words}, indent=1))
    rgb = page_rgb(str(path), path.stat().st_mtime)
    n_lines = len({w["line"] for w in words})
    if joins:
        st.caption(f"{len(joins)} join(s) made by hand on this page")
    if ctx is not None:
        kraken_click_view(dict(ctx, joins_path=str(joins_path)), key, words)
        if not st.toggle("Show the cut lines too (word order, gallery)", False,
                         key=f"kw_more_{key}"):
            return
    numbers = st.checkbox("Number the words", True, key=f"kw_num_{key}")
    st.image(draw_kraken_words(rgb, K["json"], words, numbers),
             caption=f"{len(words)} words on {n_lines} Kraken lines · each line is cut into "
                     f"slices at the white space · orange = baseline · saved to {out.name}",
             **FULL)
    st.download_button("Download words (JSON)", out.read_text(), out.name,
                       "application/json", key=f"kw_dl_{key}")
    with st.expander("Words line by line"):
        by_line: dict[str, list[dict]] = {}
        for w in words:
            by_line.setdefault(w["line"], []).append(w)
        for lid, ws in by_line.items():
            st.caption(f"line {lid} · {len(ws)} words · slant {ws[0]['slant']:.0f}° · "
                       f"x-height {ws[0]['x_height']:.0f}px")
            cols = st.columns(min(len(ws), 8))
            for k, w in enumerate(ws):
                cols[k % len(cols)].image(word_crop(rgb, w), caption=w["id"].rsplit("_", 1)[1],
                                          **FULL)


# =========================================================================== #
# Clicking Kraken words into clusters
# =========================================================================== #
try:                                             # optional: pip install streamlit-image-coordinates
    from streamlit_image_coordinates import streamlit_image_coordinates as image_click
except ImportError:
    image_click = None


def img_key(word: dict) -> str:
    """Which page image a word's box refers to: the segmented page of the
    spotting (default) or the original page ('orig:<page>', Kraken words)."""
    return word.get("img", word["page"])


@st.cache_data(show_spinner=False)
def seg_to_orig(seg_path: str, orig_path: str, mtime: float) -> list[list[float]]:
    """Affine map from the segmented page the spotting boxes refer to, to
    the original page Kraken works on. Identity if the two images are the
    same; otherwise found from matching ORB features (RANSAC), so a crop,
    scale or deskew in segment() is followed; size ratio as a last resort."""
    a = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.cvtColor(np.array(Image.open(orig_path).convert("RGB")), cv2.COLOR_RGB2GRAY)
    if a.shape == b.shape and np.abs(a.astype(np.int16) - b).mean() < 10:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    orb = cv2.ORB_create(6000)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is not None and db is not None:
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
        if len(m) >= 10:
            pa = np.float32([ka[x.queryIdx].pt for x in m])
            pb = np.float32([kb[x.trainIdx].pt for x in m])
            A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC,
                                                 ransacReprojThreshold=4.0)
            if A is not None and inl is not None and inl.sum() >= 10:
                return A.tolist()
    return [[b.shape[1] / a.shape[1], 0.0, 0.0], [0.0, b.shape[0] / a.shape[0], 0.0]]


def map_box(box, A) -> list[int]:
    """Bounding box of a box's corners after the affine map A."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], float)
    q = pts @ np.asarray(A, float).T
    return [int(q[:, 0].min()), int(q[:, 1].min()), int(np.ceil(q[:, 0].max())),
            int(np.ceil(q[:, 1].max()))]


def word_bbox(w: dict) -> list[int]:
    """Bounding box of a Kraken word (its slice polygon, else its box)."""
    if len(w.get("polygon") or []) >= 3:
        p = np.array(w["polygon"])
        return [int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())]
    return list(w["box"])


def clustered_on_page(ann: Annotation, words: dict, page: str, A) -> tuple[list[str], np.ndarray]:
    """Active words of open clusters on one page, with their boxes in the
    original page's coordinates."""
    ids, boxes = [], []
    for wid, v in ann.data["words"].items():
        c = ann.clusters.get(v["cluster"])
        word = words.get(wid)
        if v["removed"] or c is None or c["status"] not in OPEN or word is None \
                or word["page"] != page:
            continue
        ids.append(wid)
        boxes.append(word["box"] if img_key(word).startswith("orig:") else map_box(word["box"], A))
    return ids, np.array(boxes, float).reshape(-1, 4)


def coverage_of(kwords: list[dict], ids: list[str], boxes: np.ndarray,
                min_overlap: float = 0.4) -> dict[str, str | None]:
    """Kraken word id -> the clustered word covering it (overlap >= min_overlap
    of the smaller box), or None if no cluster has it."""
    out = {}
    for w in kwords:
        x0, y0, x1, y1 = word_bbox(w)
        if not len(ids):
            out[w["id"]] = None
            continue
        iw = np.clip(np.minimum(x1, boxes[:, 2]) - np.maximum(x0, boxes[:, 0]), 0, None)
        ih = np.clip(np.minimum(y1, boxes[:, 3]) - np.maximum(y0, boxes[:, 1]), 0, None)
        area_k = max((x1 - x0) * (y1 - y0), 1)
        area_a = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1)
        frac = iw * ih / np.minimum(area_k, area_a)
        k = int(frac.argmax())
        out[w["id"]] = ids[k] if frac[k] >= min_overlap else None
    return out


def hog(img: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients in NumPy (no cv2.HOGDescriptor, which
    some OpenCV builds lack): unsigned orientations in `bins` bins per
    cell x cell cell, each 2x2 block of cells L2-normalised (clipped at 0.2,
    renormalised), all blocks concatenated."""
    g = img.astype(np.float32)
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    b = np.minimum((ang / (180.0 / bins)).astype(int), bins - 1)
    ch, cw = g.shape[0] // cell, g.shape[1] // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    rows, cols = np.indices(g.shape)
    ok = (rows < ch * cell) & (cols < cw * cell)
    np.add.at(hist, (rows[ok] // cell, cols[ok] // cell, b[ok]), mag[ok])
    blocks = []
    for i in range(ch - 1):
        for j in range(cw - 1):
            v = hist[i:i + 2, j:j + 2].ravel()
            v = np.minimum(v / (np.linalg.norm(v) + 1e-6), 0.2)
            blocks.append(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate(blocks)


def shape_feature(img: np.ndarray, box) -> tuple[np.ndarray, float] | None:
    """Quick shape descriptor of a word for cluster suggestions: the crop
    binarised, tightened to its ink, resized to 96x32, HOG; plus its aspect
    ratio (width / height of the ink)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    crop = img[max(y0, 0):y1, max(x0, 0):x1]
    if crop.size == 0:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(ink)
    if len(xs) < 5:
        return None
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = hog(cv2.resize(ink, (96, 32), interpolation=cv2.INTER_AREA))
    return f / (np.linalg.norm(f) + 1e-8), ink.shape[1] / max(ink.shape[0], 1)


@st.cache_data(show_spinner=False)
def member_features(paths: tuple, items: tuple) -> dict:
    """shape_feature of many words at once (cached): paths = ((image key,
    path), ...), items = ((word id, image key, box), ...)."""
    imgs = {k: page_rgb(p, Path(p).stat().st_mtime) for k, p in paths}
    out = {}
    for wid, key, box in items:
        f = shape_feature(imgs[key], box)
        if f is not None:
            out[wid] = f
    return out


def suggest_clusters(ann: Annotation, words: dict, img_paths: dict, feat,
                     per_cluster: int = 12, top: int = 6) -> list[tuple[str, float]]:
    """Open clusters ranked by how much their members look like the word
    (mean HOG cosine of up to `per_cluster` members, comparable widths)."""
    if feat is None:
        return []
    f, aspect = feat
    items = []
    for cid, c in ann.clusters.items():
        if c["status"] in OPEN:
            for w in ann.members(cid)[:per_cluster]:
                if w in words and img_key(words[w]) in img_paths:
                    items.append((w, img_key(words[w]), tuple(words[w]["box"])))
    F = member_features(tuple(sorted(img_paths.items())), tuple(items))
    scores = []
    for cid, c in ann.clusters.items():
        if c["status"] not in OPEN:
            continue
        fs = [F[w] for w in ann.members(cid)[:per_cluster] if w in F]
        fs = [g for g, a in fs if max(a / aspect, aspect / a) <= 1.6]
        if fs:
            scores.append((cid, float(np.mean(np.stack(fs) @ f))))
    return sorted(scores, key=lambda s: -s[1])[:top]


def draw_word_status(rgb: np.ndarray, kwords: list[dict], cov: dict, ann: Annotation,
                     words: dict, selected: str | None, only_missing: bool,
                     height: int = 2000) -> tuple[np.ndarray, float]:
    """The page with every Kraken word slice: tinted in its cluster's colour
    and numbered with the cluster when a cluster has it, outlined in red
    when no cluster has it; the selected word in yellow. Returns the image
    and its scale (image px per page px)."""
    s = height / rgb.shape[0]
    canvas = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    tint = canvas.copy()
    order = [cid for cid, _ in sorted(ann.clusters.items(), key=lambda x: x[1]["number"])]
    colours = dict(zip(order, palette(len(order))))
    marks = []
    for w in kwords:
        wid = cov.get(w["id"])
        if only_missing and wid is not None:
            continue
        pts = (np.array(w["polygon"] if len(w.get("polygon") or []) >= 3 else w["quad"])
               * s).round().astype(np.int32)
        if wid is None:
            cv2.fillPoly(tint, [pts], (255, 90, 90))
            marks.append((pts, (220, 0, 0), 2, None))
        else:
            cid = ann.data["words"][wid]["cluster"]
            cv2.fillPoly(tint, [pts], colours[cid])
            marks.append((pts, colours[cid], 1, str(ann.clusters[cid]["number"])))
        if w["id"] == selected:
            marks.append((pts, (255, 200, 0), 4, None))
    canvas = cv2.addWeighted(tint, 0.25, canvas, 0.75, 0)
    for pts, col, th, label in marks:
        cv2.polylines(canvas, [pts], True, col, th)
        if label:
            cv2.putText(canvas, label, (int(pts[:, 0].min()), int(pts[:, 1].min()) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return canvas, s


def word_at(kwords: list[dict], x: float, y: float) -> str | None:
    """The Kraken word whose slice contains the point (page coordinates);
    if none, the nearest one within a few pixels."""
    best, best_d = None, 8.0
    for w in kwords:
        pts = np.array(w["polygon"] if len(w.get("polygon") or []) >= 3 else w["quad"],
                       np.float32)
        d = cv2.pointPolygonTest(pts, (float(x), float(y)), True)
        if d >= 0:
            return w["id"]
        if -d < best_d:
            best, best_d = w["id"], -d
    return best


def go_to_cluster(stem: str, cid: str) -> None:
    """Select a cluster in the sidebar (shown in the 'Annotate clusters' tab)."""
    st.session_state["cluster_select"] = cid
    flash("ok", f"cluster #{get_ann(stem).clusters[cid]['number']} selected - open the "
                f"'Annotate clusters' tab (sidebar filter: all)")


def word_panel(ctx: dict, page: str, w: dict, cov: dict, A) -> None:
    """Right-hand panel for the selected Kraken word: what it is, whether a
    cluster has it, and - if none does - add it to a suggested cluster or
    start a new cluster with it."""
    ann, words, rgb, who = ctx["ann"], ctx["words"], ctx["rgb"], ctx["who"]
    orig = rgb[f"orig:{page}"]
    st.image(word_crop(orig, w), caption=f"line {w['line']} · word {w['id'].rsplit('_w', 1)[1]}",
             **FULL)
    line_words = [v for v in ctx["kwords"] if v["line"] == w["line"]]
    k = next(i for i, v in enumerate(line_words) if v["id"] == w["id"])
    jp, jn = st.columns(2)
    jp.button("⟵ Join with previous", key=f"jp_{w['id']}", disabled=k == 0, **FULL,
              on_click=join_words, args=(ctx["joins_path"],
                                         join_point(line_words[k - 1], w) if k else [0, 0]),
              help="the word was cut in two: undo that cut")
    jn.button("Join with next ⟶", key=f"jn_{w['id']}", disabled=k == len(line_words) - 1,
              **FULL, on_click=join_words,
              args=(ctx["joins_path"],
                    join_point(w, line_words[k + 1]) if k < len(line_words) - 1 else [0, 0]))
    if w.get("joined"):
        st.button("Undo the joins of this line", key=f"uj_{w['id']}", **FULL,
                  on_click=undo_joins, args=(ctx["joins_path"], line_words))
    wid = cov.get(w["id"])
    if wid is not None:
        cid = ann.data["words"][wid]["cluster"]
        c = ann.clusters[cid]
        st.success(f"Already in cluster #{c['number']} · {ann.label_text(c)}")
        st.button(f"Select cluster #{c['number']}", on_click=go_to_cluster, args=(ctx["stem"], cid),
                  key=f"goto_{w['id']}", **FULL)
        return
    st.warning("No cluster has this word.")
    if not who:
        st.caption("Enter your name as annotator in the sidebar to add it.")
        return
    stem = ctx["stem"]
    x0, y0, x1, y1 = word_bbox(w)
    word = dict(id=f"{page}:k:{w['id']}", page=page, img=f"orig:{page}",
                token=f"kraken {w['id']}", line=w["line"], candidate=None,
                box=[x0, y0, x1, y1], quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                polygon=w.get("polygon"),
                seg_box=map_box([x0, y0, x1, y1],                   # for the page overlay
                                cv2.invertAffineTransform(np.asarray(A, np.float64))),
                best_z=None, origin="manual")
    img_paths = ctx["img_paths"]
    sugg = suggest_clusters(ann, words, img_paths, shape_feature(orig, word["box"]))
    targets = targets_of(ann, "")
    ranked = [cid for cid, _ in sugg] + [cid for cid in targets if cid not in dict(sugg)]
    score = dict(sugg)
    if ranked:
        into = st.selectbox(
            "Add to cluster", ranked, key=f"into_{w['id']}",
            format_func=lambda cid: targets[cid] + (f"  ·  look-alike {score[cid]:.2f}"
                                                    if cid in score else ""),
            help="ordered by how much the cluster's words look like this one")
        with st.expander(f"Words of {targets[into].split(' ')[0]}"):
            mem = ann.members(into)[:12]
            cols = st.columns(3)
            for i, m in enumerate(mem):
                if m in words:
                    cols[i % 3].image(word_image(rgb, words[m], {}, False), **FULL)
        st.button("Add to this cluster", type="primary", key=f"addto_{w['id']}", **FULL,
                  on_click=do, args=(stem, f"word added to {targets[into].split(' ')[0]}",
                                     "add_word", word, into, who))
    st.button("New cluster with this word", key=f"addnew_{w['id']}", **FULL, on_click=do,
              args=(stem, "new cluster started with this word", "add_word", word, None, who))


def kraken_click_view(ctx: dict, page: str, kwords: list[dict]) -> None:
    """The page as Kraken word slices coloured by cluster (red = no cluster);
    clicking a slice (or picking it from the list) opens its panel."""
    ann, words, rgb = ctx["ann"], ctx["words"], ctx["rgb"]
    A = seg_to_orig(ctx["seg_paths"][page], ctx["orig_paths"][page],
                    Path(ctx["orig_paths"][page]).stat().st_mtime)
    ids, boxes = clustered_on_page(ann, words, page, A)
    cov = coverage_of(kwords, ids, boxes)
    missing = [w for w in kwords if cov[w["id"]] is None]
    by_id = {w["id"]: w for w in kwords}
    sel_key = f"kw_sel_{page}"

    st.caption(f"{len(kwords)} words · {len(kwords) - len(missing)} in a cluster · "
               f"**{len(missing)} not in any cluster (red)** · click a word to add it")
    only_missing = st.toggle("Show only words without a cluster", False,
                             key=f"kw_only_{page}")
    left, right = st.columns([3, 1], gap="large")
    with left:
        img, s = draw_word_status(rgb[f"orig:{page}"], kwords, cov, ann, words,
                                  st.session_state.get(sel_key), only_missing)
        if image_click is not None:
            click = image_click(Image.fromarray(img), key=f"kw_click_{page}",
                                use_column_width="always")
            if click and click.get("width"):
                scale = img.shape[1] / click["width"]
                hit = word_at(kwords, click["x"] * scale / s, click["y"] * scale / s)
                if hit and (click["x"], click["y"]) != st.session_state.get(sel_key + "_xy"):
                    st.session_state[sel_key] = hit
                    st.session_state[sel_key + "_xy"] = (click["x"], click["y"])
                    st.rerun()                     # redraw with the word highlighted
        else:
            st.image(img, **FULL)
            st.caption("To click on the words: `pip install streamlit-image-coordinates`, "
                       "then restart the app. Until then, pick the word in the list.")
    with right:
        pool = missing if only_missing or image_click is None else kwords
        names = {w["id"]: f"line {w['line']} · word {w['id'].rsplit('_w', 1)[1]}"
                          + ("" if cov[w["id"]] else "  (no cluster)") for w in pool}
        if pool:
            cur = st.session_state.get(sel_key)
            pick = f"{sel_key}_pick_{cur}"             # new widget per selection: shows it
            if cur not in names:                       # nothing chosen yet: first red word
                cur = next((k for k in names if cov[k] is None), next(iter(names)))
                st.session_state[sel_key] = cur
                pick = f"{sel_key}_pick_{cur}"
            st.selectbox("Word (or click it on the page)", list(names), format_func=names.get,
                         key=pick, index=list(names).index(cur),
                         on_change=lambda: st.session_state.update(
                             {sel_key: st.session_state[pick]}))
        sel = st.session_state.get(sel_key)
        if sel in by_id:
            word_panel(dict(ctx, kwords=kwords), page, by_id[sel], cov, A)


# =========================================================================== #
# Annotation state (replaces annotation.py)
# =========================================================================== #
class Annotation:
    """Clusters, words, concepts and readings of one page, in annotation.json.

    words     word_id -> {cluster, removed}
    clusters  cid -> {number, status (pending/labelled/removed/merged), concept,
                      value, value_source (auto/manual), auto_name, auto_votes,
                      note, merged_into, annotator, updated_at}
    concepts  the concept list
    readings  word_id -> {diplomatic, expanded, confidence, cluster_check}
    history   every action: who, when, what"""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else None

    # ---- life cycle
    def start(self, words_json: dict, source: str) -> None:
        """New annotation from the clustering output (clusters numbered 1, 2, ...)."""
        clusters, words = {}, {}
        for k, (cid, members) in enumerate(words_json["clusters"].items(), 1):
            clusters[str(k)] = dict(number=k, status="pending", concept=None, value=None,
                                    value_source=None, auto_name=None, auto_votes=None,
                                    note=None, merged_into=None, annotator=None, updated_at=None)
            for m in members:
                words[m] = dict(cluster=str(k), removed=False)
        self.data = dict(source=source, created_at=now(), clusters=clusters, words=words,
                         concepts=[], readings={}, history=[])
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        os.replace(tmp, self.path)

    def log(self, who: str, action: str, cluster: str | None = None, **payload) -> None:
        self.data["history"].append(dict(ts=now(), annotator=who, action=action,
                                         cluster=cluster, **payload))

    # ---- reading
    @property
    def clusters(self) -> dict:
        return self.data["clusters"]

    def members(self, cid: str, removed: bool = False) -> list[str]:
        return [w for w, v in self.data["words"].items()
                if v["cluster"] == cid and v["removed"] == removed]

    def concepts(self) -> list[str]:
        return sorted(self.data["concepts"], key=str.casefold)

    def find_concept(self, name: str | None) -> str | None:
        key = (name or "").strip().casefold()
        return next((c for c in self.data["concepts"] if c.casefold() == key), None) if key else None

    def similar_concepts(self, name: str) -> list[str]:
        import difflib
        low = {c.casefold(): c for c in self.data["concepts"]}
        return [low[m] for m in difflib.get_close_matches(name.strip().casefold(), low, 3, 0.75)]

    def signifiers(self, concept: str) -> list[str]:
        return list(dict.fromkeys(c["value"] for c in self.clusters.values()
                                  if c["concept"] == concept and c["status"] == "labelled"
                                  and c["value"]))

    def concept_users(self, concept: str) -> list[dict]:
        return [c for c in self.clusters.values()
                if c["concept"] == concept and c["status"] != "merged"]

    def label_text(self, c: dict) -> str:
        if c["status"] == "labelled" and c["concept"]:
            return f"{c['value'] or '?'} -> {c['concept']} ({c['value_source']})"
        if c["auto_name"]:
            return f"unassigned - auto: {c['auto_name']} {c['auto_votes'] or ''}".rstrip()
        if c["auto_votes"]:
            return f"unassigned - auto: {c['auto_votes']}"
        return "unassigned"

    def progress(self) -> dict:
        st_ = Counter(c["status"] for c in self.clusters.values())
        labelled = [c for c in self.clusters.values() if c["status"] == "labelled"]
        return dict(total=len(self.clusters), **{s: st_[s] for s in
                                                 ("pending", "labelled", "removed", "merged")},
                    concepts=len({c["concept"] for c in labelled}), signifiers=len(labelled))

    # ---- actions (each raises ValueError when not allowed)
    def _open(self, cid: str) -> dict:
        c = self.clusters[cid]
        if c["status"] in ("removed", "merged"):
            raise ValueError(f"cluster #{c['number']} is {c['status']}; restore it first")
        return c

    def _touch(self, c: dict, who: str, **fields) -> None:
        c.update(fields, annotator=who, updated_at=now())

    def assign(self, cid: str, concept: str, who: str, value: str | None,
               note: str | None, new: bool) -> str:
        c = self._open(cid)
        concept = (concept or "").strip()
        if not concept:
            raise ValueError("choose a concept from the list or define a new one")
        if not self.members(cid):
            raise ValueError("this cluster has no words left")
        source = "manual" if value else None
        if not value and c["auto_name"]:
            value, source = c["auto_name"], "auto"
        if not value:
            raise ValueError("type the signifier (value of this cluster)")
        existing = self.find_concept(concept)
        if new and existing:
            raise ValueError(f"'{existing}' is already in the list - choose {EXISTING}")
        if not new and not existing:
            raise ValueError(f"'{concept}' is not in the concept list")
        if not existing:
            self.data["concepts"].append(concept)
            self.log(who, "add_concept", concept=concept)
        concept = existing or concept
        self._touch(c, who, status="labelled", concept=concept, value=value,
                    value_source=source, note=note)
        self.log(who, "assign", cid, concept=concept, value=value, source=source)
        return concept

    def unassign(self, cid: str, who: str) -> None:
        self._touch(self._open(cid), who, status="pending", concept=None, value=None,
                    value_source=None)
        self.log(who, "unassign", cid)

    def remove(self, cid: str, who: str, reason: str | None) -> None:
        self._touch(self._open(cid), who, status="removed", note=reason)
        self.log(who, "remove", cid, reason=reason)

    def restore(self, cid: str, who: str) -> None:
        c = self.clusters[cid]
        self._touch(c, who, status="labelled" if c["concept"] else "pending")
        self.log(who, "restore", cid)

    def merge(self, cid: str, into: str, who: str) -> None:
        if cid == into:
            raise ValueError("cannot merge a cluster into itself")
        self._open(cid), self._open(into)
        for w in self.data["words"].values():
            if w["cluster"] == cid:
                w["cluster"] = into
        self._touch(self.clusters[cid], who, status="merged", merged_into=into)
        self.log(who, "merge", cid, into=into)

    def move_word(self, word: str, into: str, who: str) -> None:
        src = self.data["words"][word]["cluster"]
        if src == into:
            raise ValueError("the word is already in that cluster")
        self._open(into)
        self.data["words"][word].update(cluster=into, removed=False)
        self.log(who, "move_word", src, word=word, to=into)

    def new_cluster(self, words: list[str], who: str) -> str:
        cid = str(max(int(k) for k in self.clusters) + 1)
        self.clusters[cid] = dict(number=int(cid), status="pending", concept=None, value=None,
                                  value_source=None, auto_name=None, auto_votes=None, note=None,
                                  merged_into=None, annotator=who, updated_at=now())
        for w in words:
            self.data["words"][w].update(cluster=cid, removed=False)
        self.log(who, "new_cluster", cid, words=words)
        return cid

    def add_word(self, word: dict, into: str | None, who: str) -> str:
        """Add a word no cluster had (clicked on the page) to cluster `into`,
        or to a new cluster (into=None). The word's geometry is kept in
        extra_words, so it survives re-runs of the spotting."""
        wid = word["id"]
        old = self.data["words"].get(wid)
        if old is not None and not old["removed"] and \
                self.clusters[old["cluster"]]["status"] in OPEN:
            raise ValueError(f"this word is already in cluster "
                             f"#{self.clusters[old['cluster']]['number']}")
        if into is not None:
            self._open(into)
        self.data.setdefault("extra_words", {})[wid] = word
        self.data["words"][wid] = dict(cluster=into, removed=False)
        if into is None:
            into = self.new_cluster([wid], who)
        self.log(who, "add_word", into, word=wid, box=word["box"])
        return into

    def remove_word(self, word: str, who: str, restore: bool = False) -> None:
        self.data["words"][word]["removed"] = not restore
        self.log(who, "restore_word" if restore else "remove_word",
                 self.data["words"][word]["cluster"], word=word)

    def delete_concept(self, concept: str, who: str) -> None:
        if self.concept_users(concept):
            raise ValueError(f"'{concept}' is still used - unassign its clusters first")
        self.data["concepts"].remove(concept)
        self.log(who, "delete_concept", concept=concept)

    # ---- automatic names from DeepSeek's readings
    def add_readings(self, result: dict, form: str) -> None:
        for w, v in result.get("words", {}).items():
            if v.get("diplomatic") or v.get("expanded"):
                self.data["readings"][w] = {k: v.get(k) for k in
                                            ("diplomatic", "expanded", "confidence",
                                             "cluster_check")}
        self.data["reading_form"] = form

    def reading(self, word: str) -> str | None:
        r = self.data["readings"].get(word)
        if not r or r.get("cluster_check") is False:
            return None
        form = self.data.get("reading_form", "expanded")
        return r.get(form) or r.get("expanded") or r.get("diplomatic")

    def update_auto_names(self) -> None:
        """Each open cluster's automatic name = the reading most of its current
        words agree on (votes 'k/n'); no name when two or more readings disagree."""
        for cid, c in self.clusters.items():
            if c["status"] not in OPEN:
                continue
            rd = [r for w in self.members(cid) if (r := self.reading(w))]
            if not rd:
                c.update(auto_name=None, auto_votes=None)
                continue
            name, k = Counter(rd).most_common(1)[0]
            agreed = len(rd) == 1 or k >= 2
            c.update(auto_name=name if agreed else None, auto_votes=f"{k}/{len(rd)}")


ANNOTATION_FILE = "spotting_annotation.json"   # own file: never shares app.py's annotation.json


def same_source(saved: str | None, current: str) -> bool:
    """Same spotting settings? Annotations saved by earlier versions of the app
    also hold the (now removed) coverage settings; those are ignored."""
    try:
        a, b = json.loads(saved), json.loads(current)
    except (TypeError, ValueError):
        return False
    if isinstance(a, dict):
        a.pop("coverage", None)
    return a == b


def get_ann(stem: str) -> Annotation:
    return Annotation(run_dir(stem) / ANNOTATION_FILE)


def flash(kind: str, text: str) -> None:
    st.session_state["flash"] = (kind, text)


def show_flash() -> None:
    kind, text = st.session_state.pop("flash", (None, None))
    if kind == "ok":
        st.success(text)
    elif kind == "error":
        st.error(text)


def do(stem: str, done: str, action: str, *args, **kwargs) -> None:
    """Button callback: re-read the file, run one action, save, refresh names."""
    ann = get_ann(stem)
    try:
        getattr(ann, action)(*args, **kwargs)
    except (ValueError, KeyError) as e:
        flash("error", str(e))
        return
    ann.update_auto_names()
    ann.save()
    flash("ok", done)


# =========================================================================== #
# Fit check from the spotting's own matches
# =========================================================================== #
class Check:
    def __init__(self, word, cluster, recommendation, best=None, best_number=None):
        self.word, self.cluster = word, cluster
        self.recommendation, self.best_cluster, self.best_number = recommendation, best, best_number


def load_edges(path: str) -> dict[str, dict[str, float]]:
    """word -> {matched word: z} from edges.json (every accepted match)."""
    adj: dict[str, dict[str, float]] = {}
    for key, z in json.loads(Path(path).read_text()).items():
        a, b = key.split("|")
        adj.setdefault(a, {})[b] = z
        adj.setdefault(b, {})[a] = z
    return adj


def check(ann: Annotation, T: dict | None, mu: float, sd: float,
          adj: dict[str, dict[str, float]], margin: float, fit_z: float,
          ratio: float) -> dict[str, Check]:
    """Does every active word of an open cluster fit it? As in app.py, from
    the MEAN similarity of the word to the CURRENT members of each cluster
    (whitened-HOG cosine as a z-score against random token pairs):
      own    mean z to the rest of its own cluster
      other  mean z to the best other open cluster (compatible width)
      move    other >= own + margin and other >= fit_z     (fits #N better)
      remove  own < fit_z                                  (fits no cluster)
      keep / single otherwise
    The spotting's accepted matches (adj) overrule both: no move while the
    word's strongest match to its own cluster is at least as strong as its
    strongest match to the target, and no removal while it has an accepted
    match with a member of its own cluster."""
    groups = {cid: ann.members(cid) for cid, c in ann.clusters.items() if c["status"] in OPEN}
    out: dict[str, Check] = {}
    if T is None:                                     # no features: spotting edges only
        for cid, mem in groups.items():
            for w in mem:
                out[w] = Check(w, cid, "single" if len(mem) == 1 else "keep")
        return out
    cids, sums, sizes, widths = cluster_profile(T, groups)
    col = {cid: n for n, cid in enumerate(cids)}
    for cid, mem in groups.items():
        for w in mem:
            if len(mem) == 1:
                out[w] = Check(w, cid, "single")
                continue
            if w not in T["pos"] or cid not in col:
                out[w] = Check(w, cid, "keep")
                continue
            i, n = T["pos"][w], col[cid]
            dots = T["F"][i] @ sums.T
            mean = dots / sizes
            mean[n] = (dots[n] - 1.0) / max(sizes[n] - 1, 1)     # without the word itself
            z = (mean - mu) / sd
            own_z = z[n]                                  # own cluster: no width test
            z[~width_ok(np.full(len(widths), T["width"][i]), widths, ratio)] = -np.inf
            z[n] = -np.inf
            best = int(z.argmax()) if len(z) > 1 else None
            links = adj.get(w, {})
            own_e = max((links[m] for m in mem if m != w and m in links), default=-np.inf)
            if best is not None and np.isfinite(z[best]) and z[best] >= own_z + margin \
                    and z[best] >= fit_z:
                target = cids[best]
                tgt_e = max((links[m] for m in groups[target] if m in links), default=-np.inf)
                if not (np.isfinite(own_e) and own_e >= tgt_e):
                    out[w] = Check(w, cid, "move", target, ann.clusters[target]["number"])
                    continue
            if own_z < fit_z and not np.isfinite(own_e):
                out[w] = Check(w, cid, "remove")
            else:
                out[w] = Check(w, cid, "keep")
    return out


def apply_checks(ann: Annotation, cid: str, checks: dict[str, Check], who: str) -> None:
    for w in ann.members(cid):
        k = checks.get(w)
        if k and k.recommendation == "move":
            ann.move_word(w, k.best_cluster, who)
        elif k and k.recommendation == "remove":
            ann.remove_word(w, who)


# =========================================================================== #
# Images (crops from the segmented pages the boxes refer to)
# =========================================================================== #
def palette(n: int) -> list[tuple[int, int, int]]:
    """n well-separated RGB colours (golden-ratio hues, two brightness levels)."""
    out = []
    for k in range(n):
        hue = int((k * 0.618034 % 1.0) * 179)
        val = 200 if k % 2 else 150
        hsv = np.uint8([[[hue, 255, val]]])
        out.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]))
    return out


def draw_annotation(rgb: np.ndarray, page: str, ann: Annotation, words: dict,
                    statuses: tuple[str, ...], selected: str | None = None,
                    height: int = 2000) -> tuple[np.ndarray, int]:
    """The page with every word of the CURRENT annotation boxed in its
    cluster's colour and numbered with the cluster number - spotting words,
    added words and the annotator's moves alike. Added words get a thinner
    box; the selected cluster is drawn thicker. Returns the image and the
    number of clusters drawn on this page."""
    s = height / rgb.shape[0]
    canvas = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    shown = [(cid, c) for cid, c in sorted(ann.clusters.items(), key=lambda x: x[1]["number"])
             if c["status"] in statuses]
    colours = dict(zip((cid for cid, _ in shown), palette(len(shown))))
    drawn = set()
    for cid, c in shown:
        for w in ann.members(cid):
            word = words.get(w)
            if word is None or word["page"] != page:
                continue
            x0, y0, x1, y1 = (int(v * s) for v in word.get("seg_box", word["box"]))
            added = word.get("origin", "spotting") != "spotting"
            thick = 6 if cid == selected else (2 if added else 4)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colours[cid], thick)
            cv2.putText(canvas, str(c["number"]), (x0, max(y0 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, colours[cid], 2)
            drawn.add(cid)
    return canvas, len(drawn)


def word_image(rgb: dict, word: dict, line_polys: dict, context: bool, h: int = 80):
    """The word, or with context=True its line: a band of the page as high as
    the box (plus a margin) and wide enough to show the neighbouring words,
    with the word boxed in red."""
    img = rgb[img_key(word)]
    x0, y0, x1, y1 = word["box"]
    H, W = img.shape[:2]
    if context:
        bh = y1 - y0
        cx0, cx1 = max(0, x0 - 4 * bh), min(W, x1 + 4 * bh)
        cy0, cy1 = max(0, y0 - bh // 2), min(H, y1 + bh // 2)
        crop = img[cy0:cy1, cx0:cx1].copy()
        cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (220, 0, 0), 2)
        return crop
    m = max(2, (y1 - y0) // 10)
    crop = img[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    return tc.upscale(crop, h)


# =========================================================================== #
# Sidebar: spotting settings
# =========================================================================== #
def spotting_settings() -> dict:
    """Settings passed to your CrossParams. They start from batch_processing's
    own SETTINGS["params"] (else the CrossParams defaults), so the app runs what
    batch_processing runs."""
    d = getattr(bp, "SETTINGS", {}).get("params") or bp.CrossParams()
    with st.sidebar.expander("Word spotting (batch_processing)"):
        within = st.checkbox("Similar words inside each page", True)
        cross = st.checkbox("Similar words between pages", True)
        n_jobs = st.number_input("Parallel processes", 1, os.cpu_count() or 1, 1,
                                 help="page comparisons in parallel (needs fork: Linux / macOS)")
        st.caption("**Matching** (re-matches all pages)")
        p = dict(
            z_threshold=st.slider("z threshold", 1.0, 6.0, float(d.z_threshold), 0.25,
                                  help="higher = fewer, more reliable matches"),
            hog_weight=st.slider("HOG weight", 0.0, 1.0, float(d.hog_weight), 0.05),
            dtw_weight=st.slider("DTW weight", 0.0, 1.0, float(d.dtw_weight), 0.05),
            context_weight=st.slider("Phrase-context bonus", 0.0, 1.0, float(d.context_weight),
                                     0.05, help="0 = off"),
            min_ink=st.slider("Min ink fraction", 0.0, 0.1, float(d.min_ink), 0.005),
        )
        st.caption("**Acceptance** (re-matches all pages): which words get a match at "
                   "all. Looser values find more words, and more wrong pairs.")
        p.update(
            margin=st.slider("Margin to the query's best", 0.25, 3.0, float(d.margin), 0.25,
                             help="a match must be this close (SD units) to the best "
                                  "match of both words"),
            top_k=st.slider("Reciprocal top-k (between pages)", 1, 15, int(d.top_k)),
            top_k_within=st.slider("Reciprocal top-k (within a page)", 1, 20,
                                   int(d.top_k_within)),
            score_floor=st.slider("Score floor", 0.0, 3.0, float(d.score_floor), 0.25),
            min_match_lines=st.slider("Min matchable width (line pitches)", 0.6, 2.0,
                                      float(d.min_match_lines), 0.1,
                                      help="narrower words (di, la, e ...) are never "
                                           "matched by the spotting"),
        )
        st.caption("**Clustering** (fast: matches are cached)")
        p.update(
            cluster_min_z=(p["z_threshold"] if st.checkbox(
                "Use every accepted pair for clusters", True,
                help="on: cluster min z = z threshold, so every pair in within_<page>.jpg "
                     "can build clusters (exactly as check_spotting_app.py); off: only pairs "
                     "with z >= Cluster min z (batch_processing's SETTINGS: 4.0)") else
                st.slider("Cluster min z", 1.0, 8.0, float(d.cluster_min_z or 4.0), 0.25,
                          help="only matches this strong build clusters")),
            cluster_resolution=st.slider("Louvain resolution", 0.2, 3.0,
                                         float(d.cluster_resolution), 0.1,
                                         help="higher = smaller, tighter clusters"),
            min_link_frac=st.slider("Prune: min link fraction", 0.0, 1.0,
                                    float(d.min_link_frac), 0.05),
        )
        merge = st.checkbox("Merge split clusters", d.merge_z is not None)
        p["merge_z"] = st.slider("Merge z", 1.0, 6.0, float(d.merge_z or 3.0), 0.25) if merge else None
    # Start from EVERY field of batch_processing's CrossParams (as check_spotting_app.py
    # does) and override only the fields that have a slider. Passing the sliders alone
    # would reset every other field to the CrossParams class defaults, not SETTINGS.
    full = {k: getattr(d, k) for k in SPOT_FIELDS}
    full.update({k: v for k, v in p.items() if k in SPOT_FIELDS})
    return dict(params=tuple(sorted(full.items())), within=within, cross=cross, n_jobs=int(n_jobs))


# =========================================================================== #
# DeepSeek
# =========================================================================== #
WORD_PROMPT = (
    "The images show handwritten samples of the same word from an 18th-century Italian "
    "manuscript. Recognise this word. Reply with the word only, exactly as it is written - "
    "nothing else."
)


def deepseek_settings() -> None:
    """DeepSeek settings (sidebar). The transcription itself is started per
    cluster, from the Transcribe button in the cluster's Label panel."""
    with st.sidebar.expander("DeepSeek settings"):
        st.session_state.setdefault("ds_key", os.environ.get("DEEPSEEK_API_KEY", ""))
        st.text_input("API key", type="password", key="ds_key")
        st.session_state.setdefault("ds_model", tc.MODEL)
        st.text_input("Model", key="ds_model")
        st.session_state.setdefault("ds_samples", 8)
        st.slider("Words sent per cluster", 2, 12, key="ds_samples",
                  help="the first N words of the cluster; the rest stay unread")
        st.session_state.setdefault("ds_think", False)
        st.toggle("Let the model think first", key="ds_think",
                  help="slower and more expensive; can help with hard words")
        st.session_state.setdefault("ds_prompt", WORD_PROMPT)
        st.text_area("Prompt", key="ds_prompt", height=150,
                     help="sent after the word images; the model's reply is used as the word")
        if st.session_state["ds_prompt"] != WORD_PROMPT:
            st.button("Reset prompt", on_click=lambda: st.session_state.update(
                ds_prompt=WORD_PROMPT))


def ask_word(images: list[np.ndarray], prompt: str, think: bool = False) -> str:
    """Send the word images + prompt to DeepSeek; return the word it replies with.

    DeepSeek models think before answering unless told not to, and max_tokens
    counts the thinking too: with a small limit the thinking uses it all and the
    answer comes back empty. So thinking is off by default; with think=True the
    limit is raised to leave room for both."""
    client = tc.make_client()
    content = [tc.image_part(im) for im in images] + [{"type": "text", "text": prompt}]
    resp = client.chat.completions.create(
        model=tc.MODEL, temperature=0, max_tokens=4000 if think else 300,
        messages=[{"role": "system",
                   "content": "You transcribe handwritten words. Reply with the transcribed "
                              "word only: no sentence, no quotes, no explanation."},
                  {"role": "user", "content": content}],
        extra_body={"thinking": {"type": "enabled" if think else "disabled"}})
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    if not text:
        thought = (getattr(choice.message, "reasoning_content", None) or "").strip()
        raise RuntimeError(
            f"empty answer (finish_reason: {choice.finish_reason})"
            + (f"; the model was still thinking: '{thought[-200:]}'" if thought else ""))
    return clean_word(text)


def clean_word(text: str) -> str:
    """Only the word from the model's reply, e.g.
    'The word is "Francesco".'  -> Francesco
    'Answer: Fran:co'           -> Fran:co    (the word's own punctuation is kept)
    '**Zanini**'                -> Zanini"""
    import re
    text = re.sub(r"```[a-z]*", "", text).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    line = lines[0]
    quoted = re.search(r'["“«*\'‘`]+([^"”»*\'’`]+)["”»*\'’`]+', line)
    if quoted:
        line = quoted.group(1)
    line = re.sub(r"^(?:the\s+)?(?:handwritten\s+)?(?:word|answer|transcription|reading|text)"
                  r"(?:\s+(?:is|reads|appears to be|says))?\s*[:\-–]?\s*", "", line, flags=re.I)
    return line.strip().rstrip(".,;").strip()


def transcribe_cluster(stem: str, rgb: dict, words: dict, cid: str) -> None:
    """Send this cluster's word images to DeepSeek (one request) with the prompt
    from the sidebar; the reply is stored as the reading of every word sent, and
    the cluster's automatic name follows from it."""
    ann = get_ann(stem)
    members = ann.members(cid)
    if not members:
        flash("error", "this cluster has no words left")
        return
    os.environ["DEEPSEEK_API_KEY"] = st.session_state["ds_key"]
    tc.MODEL = st.session_state["ds_model"]
    sent = members[:st.session_state["ds_samples"]]
    crops = [tc.upscale(tc.warp_quad(rgb[img_key(words[w])], np.array(words[w]["quad"], float)))
             for w in sent]
    try:
        word = ask_word(crops, st.session_state["ds_prompt"],
                        think=st.session_state.get("ds_think", False))
    except Exception as e:
        flash("error", f"DeepSeek failed: {e}")
        return
    if not word:
        flash("error", "DeepSeek returned an empty answer")
        return
    ann = get_ann(stem)
    for w in sent:
        ann.data["readings"][w] = dict(diplomatic=word, expanded=word, confidence=None,
                                       cluster_check=None)
    ann.data["reading_form"] = "expanded"
    ann.update_auto_names()
    ann.log(st.session_state.get("annotator") or "deepseek", "deepseek", cid,
            reading=word, words=sent, model=tc.MODEL)
    ann.save()
    flash("ok", f"cluster #{ann.clusters[cid]['number']} read as '{word}' "
                f"({len(sent)} word image(s) sent)")


# =========================================================================== #
# Sidebar navigation, export, cluster view
# =========================================================================== #
def sidebar(stem: str, ann: Annotation, checks: dict[str, Check]) -> str | None:
    sb = st.sidebar
    p = ann.progress()
    done = p["labelled"] + p["removed"] + p["merged"]
    sb.progress(done / max(p["total"], 1),
                text=f"{done}/{p['total']} decided · {p['concepts']} concepts · "
                     f"{p['signifiers']} signifiers · {p['removed']} removed")
    sb.toggle("Show words in their line", value=False, key="context")
    sb.slider("Words per row", 2, 6, 3, key="per_row")
    show = sb.radio("Show clusters", ["pending", "all", "labelled", "removed", "with flags"],
                    horizontal=True)

    flags = Counter(k.cluster for k in checks.values() if k.recommendation in ("move", "remove"))
    clusters = [(cid, c) for cid, c in sorted(ann.clusters.items(), key=lambda x: x[1]["number"])
                if c["status"] != "merged"]
    if show == "with flags":
        clusters = [(cid, c) for cid, c in clusters if flags.get(cid)]
    elif show != "all":
        clusters = [(cid, c) for cid, c in clusters if c["status"] == show]
    if not clusters:
        sb.warning("No clusters to show with this filter.")
        return None

    ids = [cid for cid, _ in clusters]
    if st.session_state.get("cluster_select") not in ids:
        st.session_state["cluster_select"] = ids[0]
    pos = ids.index(st.session_state["cluster_select"])

    def select(cid: str) -> None:
        st.session_state["cluster_select"] = cid

    a, b = sb.columns(2)
    a.button("◀ Previous", disabled=pos == 0, on_click=select, args=(ids[max(pos - 1, 0)],), **FULL)
    b.button("Next ▶", disabled=pos == len(ids) - 1, on_click=select,
             args=(ids[min(pos + 1, len(ids) - 1)],), **FULL)
    labels = {cid: f"#{c['number']} · {c['status']} · {len(ann.members(cid))} words"
                   f"{' · ⚠ ' + str(flags[cid]) if flags.get(cid) else ''} · {ann.label_text(c)}"
              for cid, c in clusters}
    sb.selectbox("Cluster", ids, key="cluster_select", format_func=labels.get)
    return st.session_state["cluster_select"]


def export_panel(stem: str, ann: Annotation, words: dict) -> None:
    with st.sidebar.expander("Export"):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["word_id", "page", "line", "cluster", "status", "concept", "value", "value_source",
                    "auto_name", "reading_diplomatic", "reading_expanded", "confidence",
                    "x0", "y0", "x1", "y1", "annotator", "updated_at", "note"])
        n = 0
        for wid, v in ann.data["words"].items():
            if v["removed"]:
                continue
            c = ann.clusters[v["cluster"]]
            if c["status"] != "labelled":
                continue
            r = ann.data["readings"].get(wid, {})
            q = np.array(words[wid]["quad"])
            w.writerow([wid, words[wid]["page"], words[wid]["line"], c["number"], c["status"], c["concept"],
                        c["value"], c["value_source"], c["auto_name"], r.get("diplomatic"),
                        r.get("expanded"), r.get("confidence"), *q.min(0).astype(int),
                        *q.max(0).astype(int), c["annotator"], c["updated_at"], c["note"]])
            n += 1
        st.download_button(f"CSV ({n} labelled words)", buf.getvalue(), f"labels_{stem}.csv",
                           "text/csv", **FULL)
        st.download_button("JSON (full annotation)", json.dumps(ann.data, indent=1,
                                                                ensure_ascii=False),
                           f"annotation_{stem}.json", "application/json", **FULL)


# =========================================================================== #
# Cluster view
# =========================================================================== #
def verdict_text(k: Check | None) -> str:
    if k is None:
        return ""
    icon, text = VERDICT[k.recommendation]
    return f"{icon} {text.format(n=k.best_number)}"


def origin_tag(word: dict) -> str:
    """Marks words the spotting did not cluster itself (added by hand on the
    Kraken page)."""
    return "" if word.get("origin", "spotting") == "spotting" else " · ➕ added by hand"


def targets_of(ann: Annotation, exclude: str) -> dict[str, str]:
    return {cid: f"#{c['number']} {c['value'] or c['auto_name'] or ''}".strip()
            for cid, c in sorted(ann.clusters.items(), key=lambda x: x[1]["number"])
            if c["status"] in OPEN and cid != exclude}


def reading_line(ann: Annotation, wid: str) -> None:
    """The LLM's reading of a word, shown right under its image."""
    r = ann.data["readings"].get(wid)
    if not r:
        return
    form = ann.data.get("reading_form", "expanded")
    main = r.get(form) or r.get("expanded") or r.get("diplomatic")
    other = r.get("diplomatic") if form == "expanded" else r.get("expanded")
    extra = f" :gray[({other})]" if other and other != main else ""
    if r.get("cluster_check") is False:
        st.markdown(f"📝 ~~{main}~~ :red[the model sees a different word]")
    else:
        st.markdown(f"📝 **{main}**{extra}")


def word_tile(stem, ann, cid, wid, k, words, rgb, line_polys, targets, who) -> None:
    with st.container(border=True):
        st.image(word_image(rgb, words[wid], line_polys, st.session_state.get("context")), **FULL)
        reading_line(ann, wid)
        st.caption(f"{words[wid]['page']} · {words[wid]['token']}{origin_tag(words[wid])}")
        text = verdict_text(k)
        flagged = k is not None and k.recommendation in ("move", "remove")
        st.markdown(f"**{text}**" if flagged else text)

        ids = list(targets)
        suggested = k.best_cluster if k is not None and k.recommendation == "move" else None
        pick, move = st.columns([3, 2], vertical_alignment="bottom")
        target = pick.selectbox("Move to", ids, format_func=targets.get,
                                label_visibility="collapsed",
                                index=ids.index(suggested) if suggested in ids else 0,
                                key=f"to_{wid}_{suggested}") if ids else None
        move.button("Move", key=f"mv_{wid}", disabled=target is None,
                    type="primary" if suggested else "secondary", **FULL, on_click=do,
                    args=(stem, f"{wid} moved to {targets.get(target, '').split(' ')[0]}",
                          "move_word", wid, target, who))
        new, rm = st.columns(2)
        new.button("New cluster", key=f"new_{wid}", **FULL, on_click=do,
                   args=(stem, f"{wid} moved to a new cluster", "new_cluster", [wid], who))
        rm.button("Remove", key=f"rm_{wid}", **FULL,
                  type="primary" if k is not None and k.recommendation == "remove" else "secondary",
                  on_click=do, args=(stem, f"{wid} removed", "remove_word", wid, who))


def save_label(stem: str, c: dict, cid: str, who: str, keys: dict) -> None:
    new = st.session_state[keys["mode"]] == NEW
    concept = (st.session_state.get(keys["new"] if new else keys["concept"]) or "").strip()
    value = st.session_state[keys["value"]].strip() or None
    note = st.session_state[keys["note"]].strip() or None
    if value is not None and value == c["auto_name"] and c["value_source"] != "manual":
        value = None                                   # automatic value accepted
    do(stem, f"cluster #{c['number']} is now a signifier of '{concept}'",
       "assign", cid, concept, who, value, note, new)


def label_panel(stem: str, ann: Annotation, cid: str, who: str, rgb: dict,
                words: dict) -> None:
    c = ann.clusters[cid]
    if c["status"] == "removed":
        st.warning(f"This cluster was removed{': ' + c['note'] if c['note'] else ''}.")
        st.button("Restore cluster", on_click=do,
                  args=(stem, f"cluster #{c['number']} restored", "restore", cid, who))
        return
    concepts = ann.concepts()
    auto = c["auto_name"]
    tag = f"{cid}_{c['value']}_{auto}_{c['concept']}_{len(concepts)}"
    keys = {k: f"{k}_{tag}" for k in ("mode", "concept", "new", "value", "note")}
    match = c["concept"] or ann.find_concept(auto)
    similar = [] if match or not auto else ann.similar_concepts(auto)
    preset = match or (similar[0] if similar else None)

    with st.container(border=True):
        st.markdown("**Label**")
        has_key = bool(st.session_state.get("ds_key"))
        read = sum(1 for w in ann.members(cid) if w in ann.data["readings"])
        if st.button("Transcribe with LLM" if not read else "Transcribe again with LLM",
                     type="primary", key=f"llm_{cid}", disabled=not has_key, **FULL,
                     help="DeepSeek reads the words of this cluster; the reading appears "
                          "under each word and fills the signifier" if has_key
                     else "enter the DeepSeek API key in the sidebar (DeepSeek settings)"):
            with st.spinner("DeepSeek is reading this cluster…"):
                transcribe_cluster(stem, rgb, words, cid)
            st.rerun()
        if not has_key:
            st.caption("Add your DeepSeek API key under *DeepSeek settings* in the sidebar.")
        st.radio("1 · Does this cluster belong to a concept in the list?",
                 (EXISTING, NEW) if concepts else (NEW,), horizontal=True, key=keys["mode"])
        if st.session_state[keys["mode"]] == EXISTING:
            st.selectbox("Concept", concepts, key=keys["concept"],
                         index=concepts.index(preset) if preset in concepts else None,
                         placeholder="check the list…")
            chosen = st.session_state[keys["concept"]]
            if similar and not match:
                st.caption(f"spelled like the automatic value: {', '.join(similar)}")
            if chosen:
                sig = ann.signifiers(chosen)
                st.caption(f"**{chosen}** · " + (" · ".join(sig) if sig else "no signifiers yet"))
            else:
                st.caption(f"Not in the list? Choose {NEW}.")
        else:
            st.text_input("New concept", auto or "", key=keys["new"],
                          placeholder="name of the new concept")
            if ann.find_concept(st.session_state[keys["new"]]):
                st.warning(f"Already in the list - choose {EXISTING}.")
        hint = f"automatic: {auto} ({c['auto_votes']})" if auto else \
            (f"automatic: readings disagree ({c['auto_votes']})" if c["auto_votes"]
             else "no automatic value - run DeepSeek in the sidebar")
        st.text_input("2 · Signifier (value of this cluster)", c["value"] or auto or "",
                      key=keys["value"], help=hint, placeholder="type the word this cluster shows")
        st.text_input("Note", c["note"] or "", key=keys["note"])
        st.button("Save", type="primary", key=f"save_{tag}", on_click=save_label,
                  args=(stem, c, cid, who, keys))

    with st.popover("Remove cluster", **FULL):
        st.text_input("Reason", key=f"reason_{cid}")
        st.button("Confirm removal", key=f"rmc_{cid}", type="primary",
                  on_click=lambda: do(stem, f"cluster #{c['number']} removed", "remove", cid, who,
                                      st.session_state.get(f"reason_{cid}") or None))
    with st.popover("Merge into another cluster", **FULL):
        targets = targets_of(ann, cid)
        if targets:
            st.selectbox("Target cluster", list(targets), format_func=targets.get,
                         key=f"merge_{cid}")
            st.button("Confirm merge", key=f"mg_{cid}", type="primary",
                      on_click=lambda: do(stem, f"cluster #{c['number']} merged", "merge", cid,
                                          st.session_state[f"merge_{cid}"], who))
    with st.popover(f"Manage concepts ({len(concepts)})", **FULL):
        if not concepts:
            st.caption("The concept list is empty.")
        for name in concepts:
            users = ann.concept_users(name)
            t, b = st.columns([4, 1], vertical_alignment="center")
            t.markdown(f"**{name}**  \n:gray[{', '.join(ann.signifiers(name)) or 'no signifiers'}]")
            b.button("🗑", key=f"delc_{name}", disabled=bool(users),
                     help="unassign its clusters first" if users else f"delete '{name}'",
                     on_click=do, args=(stem, f"concept '{name}' deleted", "delete_concept",
                                        name, who))
    if c["status"] == "labelled":
        st.button("Unassign (back to pending)", **FULL, on_click=do,
                  args=(stem, f"cluster #{c['number']} back to pending", "unassign", cid, who))


def cluster_view(stem, ann, cid, checks, words, rgb, line_polys) -> None:
    who = st.session_state.get("annotator", "").strip()
    c = ann.clusters[cid]
    ck = {w: k for w, k in checks.items() if k.cluster == cid}
    n_flag = sum(k.recommendation in ("move", "remove") for k in ck.values())
    mem = ann.members(cid)

    st.header(f"Cluster #{c['number']}")
    pages = {words[w]["page"] for w in mem}
    st.caption(f"{c['status']} · {ann.label_text(c)} · {len(mem)} words on "
               f"{len(pages)} page(s)")
    if not who:
        st.warning("Enter your name as annotator in the sidebar to make changes.")

    left, right = st.columns([3, 1], gap="large")
    with right:
        if who:
            label_panel(stem, ann, cid, who, rgb, words)
    with left:
        editable = bool(who) and c["status"] != "removed"
        if c["status"] == "removed":
            st.info("Words of a removed cluster are shown for reference only.")
        elif n_flag and who:
            def accept():
                a = get_ann(stem)
                apply_checks(a, cid, checks, who)
                a.update_auto_names()
                a.save()
                flash("ok", "suggestions applied")
            st.button(f"Accept all {n_flag} suggestion(s) for this cluster", on_click=accept)
        order = {"move": 0, "remove": 1, "single": 2, "keep": 3}
        mem.sort(key=lambda w: order[ck[w].recommendation] if w in ck else 4)
        targets = targets_of(ann, cid)
        per_row = st.session_state.get("per_row", 3)
        for r in range(0, len(mem), per_row):
            for col, wid in zip(st.columns(per_row), mem[r:r + per_row]):
                with col:
                    if editable:
                        word_tile(stem, ann, cid, wid, ck.get(wid), words, rgb, line_polys,
                                  targets, who)
                    else:
                        st.image(word_image(rgb, words[wid], line_polys,
                                            st.session_state.get("context")),
                                 caption=f"{wid} · {verdict_text(ck.get(wid))}", **FULL)
                        reading_line(ann, wid)

        for tid in dict.fromkeys(k.best_cluster for k in ck.values() if k.recommendation == "move"):
            t = ann.clusters[tid]
            tm = ann.members(tid)
            with st.expander(f"Compare with #{t['number']} · {ann.label_text(t)} ({len(tm)} words)"):
                cols = st.columns(6)
                for i, w in enumerate(tm):
                    cols[i % 6].image(word_image(rgb, words[w], line_polys, False),
                                      caption=w, **FULL)

        removed = ann.members(cid, removed=True)
        if removed and editable:
            with st.expander(f"Removed words ({len(removed)})"):
                for w in removed:
                    i, b = st.columns([4, 1])
                    i.image(word_image(rgb, words[w], line_polys, False), caption=w, **FULL)
                    b.button("Restore", key=f"rs_{w}", on_click=do,
                             args=(stem, f"{w} restored", "remove_word", w, who),
                             kwargs={"restore": True})


# =========================================================================== #
def main() -> None:
    st.set_page_config(page_title="Word-spotting annotator", layout="wide")
    sb = st.sidebar
    sb.title("Word clusters")
    uploads = sb.file_uploader("Page image(s)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                               accept_multiple_files=True,
                               help="several pages: similar words are found within and "
                                    "across all of them")
    st.session_state.setdefault("annotator", os.environ.get("USER", ""))
    sb.text_input("Annotator", key="annotator")
    settings = spotting_settings()
    show_flash()

    if not uploads:
        st.info("Upload one or more page images in the sidebar to start.")
        return
    paths = sorted({save_upload(u) for u in uploads})
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        st.error("Several images share the same file name; rename them first.")
        return
    stem = doc_name(stems)
    mtimes = tuple(p.stat().st_mtime for p in paths)

    with st.spinner(f"Word spotting on {len(paths)} page(s) - the first run segments and "
                    f"matches every page, later runs reuse the cached matches…"):
        try:
            S = step_spotting(stem, tuple(str(p) for p in paths), settings["params"],
                              settings["within"], settings["cross"], settings["n_jobs"], mtimes)
        except Exception as e:
            st.error(f"Word spotting failed: {type(e).__name__}: {e}")
            return
    W = read_clusters(S["clusters_csv"])
    if not W["clusters"]:
        explain_no_clusters(S, stems, settings["within"], settings["cross"])
        return
    words = {w["id"]: w for w in W["words"]}
    rgb = {name: page_rgb(path, Path(path).stat().st_mtime) for name, path in S["pages"].items()}
    for p in paths:                                 # originals: Kraken words refer to them
        rgb[f"orig:{p.stem}"] = page_rgb(str(p), p.stat().st_mtime)
    sb.caption(f"{len(words)} words in {len(W['clusters'])} clusters (clusters.csv)"
               + (" · matches: " + ", ".join(
                   f"{r['page_a'] if r['type'] == 'within' else r['page_a'] + '↔' + r['page_b']}"
                   f" {r['n_matches']}" for r in S.get("summary", []))
                  if len(paths) > 1 else ""))

    # annotation: started from the clusters, kept until restarted
    source = json.dumps(dict(params=settings["params"], within=settings["within"],
                             cross=settings["cross"]), sort_keys=True, default=str)
    ann = get_ann(stem)
    if ann.data is None:
        ann.start(W, source)
    words.update(ann.data.get("extra_words", {}))   # words added by clicking on the page
    # Words the spotting output does not have (an annotation from another app or an
    # older run) cannot be shown: the file is backed up, then they are dropped, or -
    # if none of its words exist - a new annotation is started.
    missing = [w for w in ann.data["words"] if w not in words]
    if missing:
        backup = ann.path.with_name(f"{ann.path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
        backup.write_text(ann.path.read_text())
        if len(missing) == len(ann.data["words"]):
            ann.start(W, source)
            st.info(f"The saved annotation did not belong to these spotting results "
                    f"(none of its words exist), so a new one was started. The old file "
                    f"was kept as {backup.name}.")
        else:
            for w in missing:
                del ann.data["words"][w]
                ann.data["readings"].pop(w, None)
            ann.update_auto_names()
            ann.save()
            st.warning(f"{len(missing)} word(s) of the annotation are not in the current "
                       f"spotting output and were dropped (backup: {backup.name}).")

    if ann.data["source"] != source and same_source(ann.data["source"], source):
        ann.data["source"] = source                # older file: coverage settings dropped
        ann.save()
    if ann.data["source"] != source and not ann.data["history"]:
        ann.start(W, source)              # nothing annotated yet: just follow the new settings
    if ann.data["source"] != source:
        st.info(f"The spotting settings changed. The annotation keeps the "
                f"{len(ann.clusters)} clusters it started from; the spotting now finds "
                f"{len(W['clusters'])}.")
        if st.button("Restart annotation from the new clusters (discards the annotation)"):
            ann.start(W, source)
            st.rerun()

    with sb.expander("Check settings"):
        margin = st.slider("Better-cluster margin (z)", 0.0, 3.0, 0.5, 0.25,
                           help="move suggested when a word resembles another cluster this "
                                "much more than its own (mean similarity, z)")
        fit_z = st.slider("Fit threshold (z)", 0.0, 6.0, 1.5, 0.25,
                          help="removal suggested when a word resembles the rest of its "
                               "cluster less than this AND has no accepted spotting match "
                               "with any of its members")
    T = table_for(S["cands"], words)
    mu, sd = similarity_stats(T["F"]) if T is not None else (0.0, 1.0)
    checks = check(ann, T, mu, sd, load_edges(S["edges"]), margin, fit_z, WIDTH_RATIO)

    deepseek_settings()
    if len(paths) > 1:
        cross = sum(1 for cid, c in ann.clusters.items() if c["status"] != "merged"
                    and len({words[w]["page"] for w in ann.members(cid) if w in words}) > 1)
        sb.caption(f"{len(paths)} pages · {cross} cluster(s) span several pages")
    cid = sidebar(stem, ann, checks)
    export_panel(stem, ann, words)

    tab_ann, tab_pages, tab_lines = st.tabs(["Annotate clusters", "Pages: clusters",
                                             "Pages: lines & brackets (Kraken)"])
    with tab_ann:
        if cid is not None:
            cluster_view(stem, ann, cid, checks, words, rgb, {})
    with tab_pages:
        page = st.selectbox("Page", list(S["pages"]), key="page_clusters") \
            if len(S["pages"]) > 1 else next(iter(S["pages"]))
        view = st.radio("Show", ["Current clusters (all, as annotated)",
                                 "Spotting only (clusters_<page>.jpg)"],
                        horizontal=True, key="page_view")
        if view.startswith("Current"):
            which = st.multiselect("Cluster status", ["pending", "labelled", "removed"],
                                   ["pending", "labelled"], key="page_status")
            img, n = draw_annotation(rgb[page], page, ann, words, tuple(which), selected=cid)
            st.image(img, caption=f"{n} clusters on this page · thin box = added by "
                                  f"hand · thick box = cluster selected in the sidebar",
                     **FULL)
        elif Path(S["overlays"][page]).exists():
            st.image(S["overlays"][page], caption="Only the clusters batch_processing found "
                                                  "itself (clusters_<page>.jpg)", **FULL)
    with tab_lines:
        page = st.selectbox("Page", stems, key="page_lines") if len(stems) > 1 else stems[0]
        path = next(p for p in paths if p.stem == page)
        if st.button("Run Kraken on this page", key=f"kraken_{page}"):
            st.session_state[f"kraken_{page}_on"] = True
        if st.session_state.get(f"kraken_{page}_on"):
            with st.spinner("Kraken (about a minute per page on CPU the first time)…"):
                try:
                    K = step_lines(str(path), path.stat().st_mtime)
                except Exception as e:
                    st.error(f"Kraken failed: {e}")
                    K = None
            if K is not None:
                view = st.radio("Show", ["Words (lines cut at white space)",
                                         "Lines and brackets"],
                                horizontal=True, key=f"kview_{page}")
                if view.startswith("Words"):
                    who = st.session_state.get("annotator", "").strip()
                    ctx = dict(stem=stem, ann=ann, words=words, rgb=rgb, who=who,
                               seg_paths=S["pages"],
                               orig_paths={p.stem: str(p) for p in paths},
                               img_paths={**{k: v for k, v in S["pages"].items()},
                                          **{f"orig:{p.stem}": str(p) for p in paths}})
                    kraken_word_view(path, K, page, ctx)
                else:
                    st.image(K["overlay"], caption="Blue = lines, dark blue = in a bracket "
                                                   "group, green = labels, red = brackets",
                             **FULL)
        else:
            st.caption("Lines, bracket groups and labels from Kraken (optional, slow).")


if __name__ == "__main__":        # streamlit runs the file as __main__; importing it
    main()                        # (e.g. from check_spotting_app.py) does not start the UI