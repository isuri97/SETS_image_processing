"""Split photographs of double-page spreads into single, cleaned pages.

For every photograph in a folder:

  1. sheet   find the paper against the desk and its four corners;
  2. warp    straighten the sheet to an upright rectangle (removes the desk,
             the tilt and the perspective of a hand-held photograph);
  3. fold    find the fold between the two pages and cut there;
  4. clean   flatten the lighting of each page, so the paper becomes even
             white and the ink stays dark (shadows, the darker fold and
             uneven yellowing are removed);
  5. save    <stem>_L.png and <stem>_R.png, plus one row per page in
             pages.csv with the transform back to the original photograph.

A photograph that holds a single page (portrait shape, or no clear fold) is
saved whole, as <stem>_P.png.

Usage
-----
    python split_pages.py /path/to/photos                 # writes /path/to/photos/pages/
    python split_pages.py /path/to/photos --out /path/to/pages
    python split_pages.py /path/to/photos --gray          # grey pages instead of colour
    python split_pages.py /path/to/photos --debug         # also save a check image per photo

Run without arguments (e.g. the Run button in PyCharm) to use SETTINGS at the
bottom of this file.

Going back to the original photograph
-------------------------------------
pages.csv stores, for each page, the 3x3 matrix H that maps a point (x, y) of
the page image to the original photograph. `to_original` applies it:

    from split_pages import load_index, to_original
    index = load_index("pages/pages.csv")
    to_original(index["img3_L"], 120, 455)   # -> (x, y) in img3.jpg

so boxes found on the split pages (e.g. symbols.csv from batch.py) can still
be traced back to the photograph. Photographs are read with cv2.imread's
default flags (EXIF orientation applied), as in the mark detector.

Images are processed one at a time and released before the next is read.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
INDEX_COLUMNS = ["page", "source_file", "source_width", "source_height", "side",
                 "page_width", "page_height", "fold_found", "to_original"]


# --------------------------------------------------------------------------
# 1. sheet
# --------------------------------------------------------------------------
def sheet_mask(bgr: np.ndarray, work: int = 1000) -> np.ndarray:
    """Boolean mask of the sheet, at the resolution of `bgr`.

    The desk is far darker than the paper, even where the paper lies in a
    shadow, so the threshold is set between the two levels measured on the
    photograph: the desk (10th percentile of brightness) and well-lit paper
    (70th percentile). Shaded paper takes on a warm tint, so colour only
    rules out strongly coloured areas (a wooden desk). The test runs on a
    reduced copy (`work` pixels wide) for speed. A wide opening removes thin
    things attached to the sheet, such as a paper tag or a ruler, and holes
    (text, stains) are filled.
    """
    h, w = bgr.shape[:2]
    s = work / w
    small = cv2.resize(bgr, (work, int(round(h * s))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v, sat = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    desk, paper = np.percentile(v, 10), np.percentile(v, 70)
    raw = ((v >= desk + 0.4 * (paper - desk)) & (sat < 120)).astype(np.uint8)
    k = max(15, int(0.04 * work)) | 1
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n < 2:
        return np.ones((h, w), bool)
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(raw)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def sheet_corners(mask: np.ndarray) -> np.ndarray:
    """The four corners of the sheet: top-left, top-right, bottom-right, bottom-left.

    The outline is simplified to a quadrilateral when possible; otherwise the
    extreme points of x + y and x - y are used, which suits a sheet whose
    corners are rounded or curled.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    for eps in (0.01, 0.02, 0.03, 0.05):
        quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(quad) == 4:
            return _order(quad.reshape(4, 2).astype(np.float32))
    pts = c.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]], np.float32)


def _order(p: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    s, d = p.sum(1), p[:, 0] - p[:, 1]
    return np.array([p[s.argmin()], p[d.argmax()], p[s.argmax()], p[d.argmin()]], np.float32)


# --------------------------------------------------------------------------
# 2. warp
# --------------------------------------------------------------------------
def straighten(bgr: np.ndarray, corners: np.ndarray, inset: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """Warp the sheet to an upright rectangle.

    The rectangle has the mean width and height of the sheet's sides. A thin
    margin (`inset` of the size) is cut so that no desk shows along the
    edges. Returns the warped image and the 3x3 matrix mapping its pixels
    back to the photograph.
    """
    tl, tr, br, bl = corners
    w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    dx, dy = inset * w, inset * h
    dst = np.array([[-dx, -dy], [w - 1 + dx, -dy], [w - 1 + dx, h - 1 + dy], [-dx, h - 1 + dy]], np.float32)
    to_sheet = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(bgr, to_sheet, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped, np.linalg.inv(to_sheet)


# --------------------------------------------------------------------------
# 3. fold
# --------------------------------------------------------------------------
@dataclass
class Fold:
    x: int  # column of the fold in the straightened sheet
    found: bool  # False if no clear fold: the sheet is treated as one page
    score: float


def find_fold(sheet: np.ndarray, lo: float = 0.38, hi: float = 0.62, min_score: float = 2.5) -> Fold:
    """Column of the fold between the two pages.

    Three cues are measured per column in the middle part of the sheet
    (`lo` to `hi` of its width), each robust to the text because it is a
    median over all rows:

      shadow  how much darker the column is than the paper 1-2 % of the
              width on either side (the fold casts a narrow shadow line);
      step    the change in paper tone across the column (the two leaves
              are often lit, or yellowed, differently);
      ink     how little writing there is around the column (the binding
              margin is empty; a blank text column is not narrow enough to
              win alone, because shadow and step are absent there).

    Each cue is turned into a z-score within the search band and summed; the
    fold is the best column. If its score is below `min_score` there is no
    clear fold and the sheet is kept whole.
    """
    h, w = sheet.shape[:2]
    gray = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = slice(int(0.05 * h), int(0.95 * h))  # top and bottom edges are often curled
    # Paper tone of each column: the median over its rows. Writing covers only
    # a small share of any column, so the median is the paper, and a thin fold
    # line (a few pixels, as on a flat-pressed sheet) is kept.
    col = gaussian_filter1d(np.median(gray[rows], axis=0), 1.0)
    d_wide = max(3, int(0.012 * w))  # a broad shadow in a curved binding
    d_thin = max(2, int(0.003 * w))  # a sharp crease
    xs = np.arange(d_wide, w - d_wide)
    shadow_wide = (col[xs - d_wide] + col[xs + d_wide]) / 2 - col[xs]
    shadow_thin = (col[xs - d_thin] + col[xs + d_thin]) / 2 - col[xs]
    step = np.abs(col[xs + d_wide] - col[xs - d_wide])
    # writing: darker than the paper of its own column
    ink = ((col[None, :] - gray[rows]) > 30).mean(axis=0)
    ink = gaussian_filter1d(ink, 0.01 * w)[xs]
    band = (xs >= lo * w) & (xs <= hi * w)

    def z(v: np.ndarray) -> np.ndarray:
        b = v[band]
        return (v - np.median(b)) / (1.4826 * np.median(np.abs(b - np.median(b))) + 1e-6)

    shadow = np.maximum(np.clip(z(shadow_wide), 0, None), np.clip(z(shadow_thin), 0, None))
    score = shadow + 0.5 * np.clip(z(step), 0, None) + 0.7 * np.clip(-z(ink), 0, None)
    score[~band] = -np.inf
    i = int(np.argmax(score))
    return Fold(int(xs[i]), bool(score[i] >= min_score), float(score[i]))


# --------------------------------------------------------------------------
# 4. clean
# --------------------------------------------------------------------------
def clean(page: np.ndarray, gray: bool = False, paper_level: float = 0.93) -> np.ndarray:
    """Even out the lighting so the paper is white and the ink keeps its colour.

    Each channel is divided by an estimate of the bare paper (the image with
    the writing closed over, then blurred). The result is stretched so that
    the paper sits at white and the darkest ink near black. Colour is kept by
    default because the mark detector tells pencil from ink by saturation.
    """
    h, w = page.shape[:2]
    k = max(15, int(0.02 * max(h, w))) | 1
    small = cv2.resize(page, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((k // 4 | 1, k // 4 | 1), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 8)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    flat = page.astype(np.float32) / np.maximum(bg, 1.0)  # paper ~ 1, ink < 1
    lum = flat.min(axis=2)
    black = float(np.percentile(lum, 0.2))
    out = np.clip((flat - black) / max(paper_level - black, 1e-3), 0, 1)
    out = (out * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if gray else out


# --------------------------------------------------------------------------
# one photograph
# --------------------------------------------------------------------------
def crop_matrix(x0: int) -> np.ndarray:
    """Matrix of a crop that starts at column x0 (page -> sheet)."""
    return np.array([[1, 0, x0], [0, 1, 0], [0, 0, 1]], np.float64)


def split_photo(path: Path, out: Path, gray: bool = False, overlap: float = 0.01,
                debug: bool = False) -> list[dict]:
    """Split one photograph and write its pages; return their pages.csv rows.

    `overlap` (a share of the sheet width) is kept on each side of the fold,
    so writing that runs into the binding is not cut off.
    """
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"could not read {path}")
    corners = sheet_corners(sheet_mask(bgr))
    sheet, sheet_to_photo = straighten(bgr, corners)
    hs, ws = sheet.shape[:2]
    fold = find_fold(sheet) if ws > 1.15 * hs else Fold(ws // 2, False, 0.0)
    pad = int(overlap * ws)
    parts = [("L", 0, min(ws, fold.x + pad)), ("R", max(0, fold.x - pad), ws)] if fold.found else [("P", 0, ws)]

    rows = []
    for side, x0, x1 in parts:
        name = f"{path.stem}_{side}"
        cv2.imwrite(str(out / f"{name}.png"), clean(sheet[:, x0:x1], gray))
        h = sheet_to_photo @ crop_matrix(x0)
        rows.append({
            "page": name, "source_file": str(path.resolve()),
            "source_width": bgr.shape[1], "source_height": bgr.shape[0], "side": side,
            "page_width": x1 - x0, "page_height": hs, "fold_found": fold.found,
            "to_original": json.dumps(np.round(h / h[2, 2], 8).tolist()),
        })
    if debug:
        _check_image(bgr, corners, sheet_to_photo, fold, hs, out / f"{path.stem}_check.jpg")
    return rows


def _check_image(bgr, corners, sheet_to_photo, fold: Fold, hs: int, path: Path) -> None:
    """The photograph with the detected sheet outline (green) and fold (red)."""
    img = bgr.copy()
    t = max(3, bgr.shape[1] // 400)
    cv2.polylines(img, [corners.astype(np.int32)], True, (0, 200, 0), t)
    if fold.found:
        line = cv2.perspectiveTransform(np.array([[[fold.x, 0], [fold.x, hs - 1]]], np.float32), sheet_to_photo)
        cv2.line(img, tuple(map(int, line[0, 0])), tuple(map(int, line[0, 1])), (0, 0, 230), t)
    s = 1600 / img.shape[1]
    cv2.imwrite(str(path), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# --------------------------------------------------------------------------
# going back to the photograph
# --------------------------------------------------------------------------
def load_index(path: str | Path) -> dict[str, dict]:
    """pages.csv as {page name: row}, with `to_original` as a 3x3 array."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["to_original"] = np.array(json.loads(r["to_original"]))
    return {r["page"]: r for r in rows}


def to_original(row: dict, x: float, y: float) -> tuple[float, float]:
    """Map a point of a split page to the original photograph."""
    p = row["to_original"] @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def box_to_original(row: dict, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
    """Smallest upright box in the photograph that holds a page box."""
    pts = [to_original(row, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return int(np.floor(min(xs))), int(np.floor(min(ys))), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


# --------------------------------------------------------------------------
# folder
# --------------------------------------------------------------------------
def split_folder(folder: str | Path, out: str | Path | None = None, gray: bool = False,
                 debug: bool = False) -> None:
    """Split every photograph directly inside `folder` (see module docstring)."""
    folder = Path(folder).resolve()
    out = Path(out).resolve() if out else folder / "pages"
    out.mkdir(parents=True, exist_ok=True)
    photos = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if not photos:
        raise SystemExit(f"No images found in {folder}")
    with (out / "pages.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for i, photo in enumerate(photos, 1):
            try:
                rows = split_photo(photo, out, gray, debug=debug)
                writer.writerows(rows)
                fh.flush()
                kind = "two pages" if len(rows) == 2 else "one page (no fold found)"
                print(f"[{i}/{len(photos)}] {photo.name}: {kind}")
            except Exception as exc:  # one bad photograph must not stop the folder
                print(f"[{i}/{len(photos)}] {photo.name}: FAILED - {type(exc).__name__}: {exc}")
            gc.collect()
    print(f"Pages in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="folder containing the photographs")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: <folder>/pages)")
    ap.add_argument("--gray", action="store_true", help="write grey pages instead of colour")
    ap.add_argument("--debug", action="store_true", help="also write <photo>_check.jpg with sheet and fold")
    a = ap.parse_args()
    split_folder(a.folder, a.out, a.gray, a.debug)


# --------------------------------------------------------------------------
# Run settings - used when the script is started without arguments
# --------------------------------------------------------------------------
SETTINGS = dict(
    folder="/home/isuri/PycharmProjects/SETS_image_processing/core/annotator/data2/data",  # the photographs
    out="/home/isuri/PycharmProjects/SETS_image_processing/core/annotator/data2/data",      # the split, cleaned pages
    gray=False,                                       # colour keeps pencil and ink apart
    debug=True,                                       # check images with the sheet and the fold
)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        split_folder(**SETTINGS)