"""Run the mark detectors on page photographs and write results for inspection.

Usage
-----
    python run.py page_1770.jpg page_1761.jpg --out output

For every page a folder `output/<page>/` is written with:

    overlay.jpg        the page with every detection drawn on it
    marks_only.png     the isolated marks on white (braces, rules, slashes, ...)
    text_only.png      the page with the marks painted out, for OCR
    brace_NN.png       one crop per brace: members (blue), brace (red), note (green)
    result.json        boxes, attributes and brace-to-text links

Pipeline
--------
    preprocess.load      crop to the sheet, binarise
    braces.detect_braces curly braces, at pixel level
    shape                pixel comparison of every brace against the others
    marks.detect_marks   rules, slashes, dashes, circles, line fillers
    layout.associate     lines grouped by each brace, and the note it points to
"""
from __future__ import annotations


import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from braces import Brace, detect_braces  # noqa: E402
from layout import BraceGroup, associate, gutter  # noqa: E402
from marks import detect_marks  # noqa: E402
from preprocess import Page, load  # noqa: E402
from shape import best_match, normalise  # noqa: E402
from strokes import line_spacing, stroke_width  # noqa: E402


# BGR colours, chosen to stay distinct on brown ink
COLOURS = {
    "brace": (40, 40, 220),
    "rule": (160, 60, 160),
    "slash": (0, 140, 255),
    "dash": (200, 160, 0),
    "circle": (200, 0, 200),
    "filler": (0, 170, 170),
    "member": (200, 90, 0),
    "note": (60, 150, 0),
}


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def analyse(path: str) -> dict:
    """Run every stage on one image and return the raw results."""
    page = load(path)
    ls, sw = line_spacing(page.ink), stroke_width(page.ink)
    rejected: list = []
    braces = detect_braces(page.ink, ls, sw, debug=rejected)
    brace_mask = np.zeros_like(page.ink)
    for b in braces:
        brace_mask |= b.mask
    gx = gutter(page.ink, ls)
    marks = detect_marks(page, brace_mask, gx, ls, sw)
    mark_mask, filler_mask = np.zeros_like(page.ink), np.zeros_like(page.ink)
    for m in marks:
        mark_mask |= m.mask
        if m.kind == "filler":
            filler_mask |= m.mask
    groups = associate(braces, page.ink, ls, sw, exclude=mark_mask & ~filler_mask, fillers=filler_mask)
    return {
        "page": page, "ls": ls, "sw": sw, "gutter": gx, "braces": braces, "marks": marks,
        "groups": groups, "brace_mask": brace_mask, "mark_mask": mark_mask,
        "rejected": [(kind, f, mask) for kind, ok, f, mask, _ in rejected if not ok],
    }


def shape_scores(braces: list[np.ndarray], rejected: list[np.ndarray],
                 references: list[np.ndarray] | None = None) -> tuple[list[float], list[float]]:
    """Pixel comparison check on size-normalised shapes (see `shape.py`).

    Returns the chamfer distance of every brace, and of every rejected tall
    stroke, to its closest reference brace. Without `references` the braces
    of the same page are used, leaving out the brace being scored. Pixel
    comparison separates braces from look-alikes if the first list is clearly
    lower than the second.
    """
    if references is None:
        own = [best_match(s, braces[:i] + braces[i + 1 :]) for i, s in enumerate(braces)] if len(braces) > 1 else []
        references = braces
    else:
        own = [best_match(s, references) for s in braces] if references else []
    others = [best_match(s, references) for s in rejected] if references else []
    return own, others


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _tint(img: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], grow: int = 1) -> None:
    """Paint `mask` in `colour` on `img`, in place, slightly thickened for visibility."""
    if grow:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    img[mask] = colour


def _label(img: np.ndarray, text: str, x: int, y: int, colour: tuple[int, int, int], scale: float = 0.6) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = max(0, min(x, img.shape[1] - tw - 4)), max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2, cv2.LINE_AA)


def draw_overlay(res: dict, first_id: int = 1, legend: bool = True) -> np.ndarray:
    """The page, lightened, with every detection drawn in its class colour."""
    page: Page = res["page"]
    img = cv2.addWeighted(page.bgr, 0.55, np.full_like(page.bgr, 255), 0.45, 0)
    for m in res["marks"]:
        _tint(img, m.mask, COLOURS[m.kind])
    for i, g in enumerate(res["groups"], first_id):
        for line in g.borderline:
            x, y, w, h = line.box
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOURS["member"], 1, cv2.LINE_4)
        for line in g.members:
            x, y, w, h = line.box
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOURS["member"], 3)
        if g.note_box:
            x, y, w, h = g.note_box
            cv2.rectangle(img, (x, y), (x + w, y + h), COLOURS["note"], 3)
        _tint(img, g.brace.mask, COLOURS["brace"])
        bx, by, bw, _ = g.brace.bbox
        _label(img, f"B{i}", bx + bw + 4, by, COLOURS["brace"], 0.8)
    if not legend:
        return img
    counts = Counter(m.kind for m in res["marks"])
    counts["brace"] = len(res["braces"])
    x = 10
    for kind in ("brace", "member", "note", "rule", "slash", "dash", "circle", "filler"):
        text = kind if kind in ("member", "note") else f"{kind} {counts.get(kind, 0)}"
        _label(img, text, x, 28, COLOURS[kind], 0.7)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 22
    return img


def draw_marks_only(res: dict) -> np.ndarray:
    """All marks on a white page, each in its class colour."""
    img = np.full_like(res["page"].bgr, 255)
    for m in res["marks"]:
        _tint(img, m.mask, COLOURS[m.kind], grow=0)
    for b in res["braces"]:
        _tint(img, b.mask, COLOURS["brace"], grow=0)
    return img


def draw_text_only(res: dict) -> np.ndarray:
    """The page with every mark painted out (inpainted from the surrounding paper)."""
    remove = cv2.dilate((res["brace_mask"] | res["mark_mask"]).astype(np.uint8), np.ones((7, 7), np.uint8))
    return cv2.inpaint(res["page"].bgr, remove, 5, cv2.INPAINT_TELEA)


def brace_crop(res: dict, g: BraceGroup, index: int, pad: int = 20) -> np.ndarray:
    """A close-up of one brace with its member lines and its note."""
    boxes = [g.brace.bbox] + [t.box for t in g.members + g.borderline + g.note]
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(res["page"].bgr.shape[1], max(b[0] + b[2] for b in boxes) + pad)
    y1 = min(res["page"].bgr.shape[0], max(b[1] + b[3] for b in boxes) + pad)
    fillers = [m for m in res["marks"] if m.kind == "filler"]
    one = {**res, "groups": [g], "marks": fillers, "braces": [g.brace]}
    return draw_overlay(one, first_id=index, legend=False)[y0:y1, x0:x1].copy()


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def to_json(res: dict, own: list[float]) -> dict:
    """Plain-data summary of one page (no masks)."""
    groups = []
    for i, g in enumerate(res["groups"]):
        groups.append({
            "id": i + 1,
            "column": g.column,
            "bbox": list(g.brace.bbox),
            "kind": g.brace.kind,
            "shape_distance": round(own[i], 2) if i < len(own) else None,
            "features": g.brace.features,
            "members": [{"box": list(t.box), "coverage": t.coverage} for t in g.members],
            "borderline": [{"box": list(t.box), "coverage": t.coverage} for t in g.borderline],
            "note": [{"box": list(t.box)} for t in g.note],
        })
    marks = [{"kind": m.kind, "bbox": list(m.bbox), **m.attrs} for m in res["marks"]]
    return {
        "line_spacing_px": int(res["ls"]), "stroke_width_px": round(float(res["sw"]), 2),
        "gutter_x": int(res["gutter"]), "braces": groups, "marks": marks,
    }


def process(path: str, out_root: Path) -> dict:
    """Analyse one page, write its output folder and return a short summary."""
    start = time.perf_counter()
    res = analyse(path)
    out = out_root / Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    brace_shapes = [normalise(b.mask) for b in res["braces"]]
    other_shapes = [normalise(mask) for _, _, mask in res["rejected"]]
    own, others = shape_scores(brace_shapes, other_shapes)
    cv2.imwrite(str(out / "overlay.jpg"), draw_overlay(res), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "marks_only.png"), draw_marks_only(res))
    cv2.imwrite(str(out / "text_only.png"), draw_text_only(res))
    for i, g in enumerate(res["groups"], 1):
        cv2.imwrite(str(out / f"brace_{i:02d}.png"), brace_crop(res, g, i))
    data = to_json(res, own)
    data["shape_check"] = {"braces": [round(v, 2) for v in own], "rejected_strokes": [round(v, 2) for v in others]}
    elapsed = time.perf_counter() - start
    data["processing_time_s"] = round(elapsed, 3)
    (out / "result.json").write_text(json.dumps(data, indent=2, default=float))
    counts = {"page": Path(path).stem, "braces": len(res["braces"]),
              "members": [len(g.members) for g in res["groups"]],
              "notes": sum(bool(g.note) for g in res["groups"]), **Counter(m.kind for m in res["marks"])}
    return {"counts": counts, "own": own, "others": others, "brace_shapes": brace_shapes, "other_shapes": other_shapes, "seconds": elapsed}


def _span(values: list[float]) -> str:
    return f"{min(values):.2f} - {max(values):.2f}  (n={len(values)})" if values else "none"


DEFAULT_DATA = Path("/home/isuri/PycharmProjects/SETS_image_processing/data2")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def expand_inputs(paths: list[str]) -> list[str]:
    """Turn a mix of image files and folders into a sorted list of image files."""
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {p}")
    return [str(f) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="*", default=[str(DEFAULT_DATA)],
                    help=f"image files or folders (default: {DEFAULT_DATA})")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    images = expand_inputs(args.images)
    if not images:
        sys.exit(f"no images found in: {', '.join(args.images)}")
    total_start = time.perf_counter()
    pages = []
    for n, path in enumerate(images, 1):
        p = process(path, Path(args.out))
        pages.append(p)
        print(f"[{n}/{len(images)}] {Path(path).name}: {p['seconds']:.2f} s")
    total = time.perf_counter() - total_start

    times = [p["seconds"] for p in pages]
    print(f"\n{len(pages)} image(s) in {total:.2f} s  "
          f"(mean {total / len(pages):.2f} s, min {min(times):.2f} s, max {max(times):.2f} s)")

    for p in pages:
        print(p["counts"])
        print("  shape distance to the other braces of the page")
        print("    braces           :", _span(p["own"]))
        print("    rejected strokes :", _span(p["others"]))
    if len(pages) > 1:
        print("cross-page check: references are the braces of all *other* pages")
        for i, p in enumerate(pages):
            refs = [s for j, q in enumerate(pages) if j != i for s in q["brace_shapes"]]
            own, others = shape_scores(p["brace_shapes"], p["other_shapes"], refs)
            print(f"  {p['counts']['page']}: braces {_span(own)} | rejected strokes {_span(others)}")


if __name__ == "__main__":
    main()