"""Detected symbols as one self-contained CSV, and their link to the spotted words.

symbols.csv has one row per detected symbol and, for every brace, one row
per text line it relates to. Every row carries what is needed to find the
symbol again later from the original photograph alone, so the file can be
read without result.json or the rest of the output folder.

Columns
-------
    source_file          path of the original photograph
    page                 its file stem (the page name used everywhere else)
    image_width/height   size of the original photograph, in pixels
    symbol               B1, B2 ... braces (as in overlay.jpg and brace_NN.png),
                         R rules, S slashes, D dashes, C circles, F fillers
    kind                 brace, rule, slash, dash, circle or filler
    relation             symbol      the mark itself
                         member      a line the brace groups (left of it)
                         borderline  a line the brace only touches
                         note        a line of the note it points to (right of it)
    line                 line number within that relation, top to bottom
    ox0, oy0, ox1, oy1   the box in the ORIGINAL photograph (x1, y1 exclusive)
    x0, y0, x1, y1       the same box in the sheet crop (text_only.png, the
                         image the word spotter reads by default)
    sheet_x, sheet_y     offset of the sheet crop in the photograph (o = crop + offset)
    column               text column index, left to right (0-3 on a double-page spread)
    column_x0/x1         the column's extent in the sheet crop
    coverage             brace lines: share of the line's x-height the brace spans
    attributes           JSON: brace type, arms, shape distance and shape features;
                         slash role, strokes and angle; circle medium; lengths
    line_spacing_px      page scale: text line pitch
    stroke_width_px      page scale: pen stroke width
    run_date             when the page was processed
    settings             JSON: layout settings of the run (pages, columns per page)

add_symbols() adds three columns to a clusters.csv, linking every word
occurrence to the brace lines its box lies on:

    symbol           e.g. "B2"            (several joined with "; ")
    symbol_relation  e.g. "member"
    symbol_line      e.g. "3"
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PREFIX = {"brace": "B", "rule": "R", "slash": "S", "dash": "D", "circle": "C", "filler": "F"}
BOX = ("x0", "y0", "x1", "y1")
OBOX = ("ox0", "oy0", "ox1", "oy1")
COLUMNS = ["source_file", "page", "image_width", "image_height",
           "symbol", "kind", "relation", "line", *OBOX, *BOX, "sheet_x", "sheet_y",
           "column", "column_x0", "column_x1", "coverage", "attributes",
           "line_spacing_px", "stroke_width_px", "run_date", "settings"]
RELATIONS = (("member", "members"), ("borderline", "borderline"), ("note", "note"))  # (relation, result.json key)
LINKED = tuple(r for r, _ in RELATIONS)  # relations through which a word is linked
LINK_COLUMNS = ["symbol", "symbol_relation", "symbol_line"]
PAGE_COLUMNS = ("page", "document", "image")  # accepted names of the page column in clusters.csv


def _plain(v):
    """JSON fallback for numpy scalars."""
    return v.item() if hasattr(v, "item") else str(v)


def _column(columns: list, x: float) -> int | str:
    """Index of the column containing x (nearest one if none does), '' if unknown."""
    if not columns:
        return ""
    for i, (a, b) in enumerate(columns):
        if a <= x < b:
            return i
    return min(range(len(columns)), key=lambda i: min(abs(x - columns[i][0]), abs(x - columns[i][1])))


# --------------------------------------------------------------------------
# symbols of one page
# --------------------------------------------------------------------------
def symbol_rows(data: dict, page: str) -> list[dict]:
    """Rows of symbols.csv from a page's result.json content (run.to_json).

    Older result.json files lack the source, size, offset and run fields;
    those cells are then left empty, and the original-photo box equals the
    sheet-crop box only if the offset is known.
    """
    dx, dy = data.get("sheet_offset", ["", ""])
    columns = data.get("columns", [])
    width, height = data.get("image_size", ["", ""])
    page_info = {
        "source_file": data.get("source_file", ""), "page": page,
        "image_width": width, "image_height": height, "sheet_x": dx, "sheet_y": dy,
        "line_spacing_px": data.get("line_spacing_px", ""), "stroke_width_px": data.get("stroke_width_px", ""),
        "run_date": data.get("run_date", ""),
        "settings": json.dumps(data["settings"]) if "settings" in data else "",
    }
    rows: list[dict] = []

    def add(symbol, kind, relation, box, line="", coverage="", attributes=None, column=None):
        x, y, w, h = (int(v) for v in box)
        col = _column(columns, x + w / 2) if column is None else column
        row = {**page_info, "symbol": symbol, "kind": kind, "relation": relation, "line": line,
               "x0": x, "y0": y, "x1": x + w, "y1": y + h, "coverage": coverage,
               "column": col,
               "column_x0": columns[col][0] if col != "" else "", "column_x1": columns[col][1] if col != "" else "",
               "attributes": json.dumps(attributes, default=_plain) if attributes else ""}
        if dx != "":
            row.update(ox0=x + dx, oy0=y + dy, ox1=x + w + dx, oy1=y + h + dy)
        rows.append(row)

    for b in data.get("braces", []):
        sid = f"B{b['id']}"
        add(sid, "brace", "symbol", b["bbox"], column=b.get("column"),
            attributes={"type": b["kind"], "arms": b.get("arms"), "shape_distance": b.get("shape_distance"),
                        "features": b.get("features", {})})
        for relation, key in RELATIONS:
            for k, t in enumerate(b.get(key, []), 1):
                add(sid, "brace", relation, t["box"], k, t.get("coverage", ""), column=b.get("column"))

    counts: dict[str, int] = {}
    for m in data.get("marks", []):
        kind = m["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        attrs = {k: v for k, v in m.items() if k not in ("kind", "bbox")}
        add(f"{PREFIX.get(kind, kind[:1].upper())}{counts[kind]}", kind, "symbol", m["bbox"], attributes=attrs)
    return rows


def write_symbols(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_symbols(results_root: Path, page: str) -> list[dict]:
    """Rows of one page: results/<page>/symbols.csv, or built from result.json
    for pages processed before symbols.csv existed. Rebuilding from result.json
    also upgrades an old-format symbols.csv to the current columns."""
    folder = Path(results_root) / page
    if (folder / "symbols.csv").exists():
        with (folder / "symbols.csv").open(newline="") as fh:
            reader = csv.DictReader(fh)
            if set(COLUMNS) <= set(reader.fieldnames or []):
                return list(reader)
    if (folder / "result.json").exists():
        return symbol_rows(json.loads((folder / "result.json").read_text()), page)
    return []


def append_symbols(path: Path, rows: list[dict]) -> int:
    """Append rows to the common symbols.csv, creating it (with its header) if needed.

    The rows are written and flushed to disk at once, so the file holds every
    finished page even if the batch stops. Returns the number of rows added.
    """
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())
    return len(rows)


def drop_pages(path: Path, pages: set[str]) -> int:
    """Remove the rows of `pages` from the common symbols.csv (they are about to be
    processed again) and bring the file to the current columns. Returns rows kept."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(newline="") as fh:
        keep = [r for r in csv.DictReader(fh) if r.get("page") not in pages]
    write_symbols(path, keep)
    return len(keep)


def merge_symbols(pages: list[str], results_root: Path, out: Path) -> int:
    """All pages' symbols in one CSV; returns the number of rows."""
    rows = [r for p in pages for r in load_symbols(results_root, p)]
    write_symbols(out, rows)
    return len(rows)


# --------------------------------------------------------------------------
# link words to symbols
# --------------------------------------------------------------------------
def _overlap(word: list[int], s: dict, keys: tuple[str, ...]) -> float:
    """Share of the word box that lies inside the symbol's line box."""
    sx0, sy0, sx1, sy1 = (int(s[k]) for k in keys)
    w = min(word[2], sx1) - max(word[0], sx0)
    h = min(word[3], sy1) - max(word[1], sy0)
    area = max(1, (word[2] - word[0]) * (word[3] - word[1]))
    return max(0, w) * max(0, h) / area


def add_symbols(clusters_csv: Path, results_root: Path, out: Path | None = None,
                original: bool = False, min_overlap: float = 0.5) -> int | None:
    """Add symbol, symbol_relation and symbol_line to every row of a clusters.csv.

    A word is linked to a brace line when at least `min_overlap` of its box
    lies inside that line's box. `original=True` when the word boxes were
    measured on the original photograph rather than the sheet crop. The file
    is rewritten in place unless `out` is given. Returns the number of linked
    word occurrences, or None if the file lacks a page column or box.
    """
    path = Path(clusters_csv)
    if not path.exists():
        return None
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    page_col = next((c for c in PAGE_COLUMNS if c in fields), None)
    if page_col is None or not set(BOX) <= set(fields):
        print(f"  symbols not added to {path}: needs a page column ({', '.join(PAGE_COLUMNS)}) "
              f"and {', '.join(BOX)}; found {', '.join(fields)}")
        return None
    keys = OBOX if original else BOX

    cache: dict[str, list[dict]] = {}
    linked = 0
    for r in rows:
        page = r[page_col]
        if page not in cache:
            cache[page] = [s for s in load_symbols(results_root, page)
                           if s["relation"] in LINKED and all(s.get(k) not in ("", None) for k in keys)]
        word = [int(float(r[c])) for c in BOX]
        hits = [s for s in cache[page] if _overlap(word, s, keys) >= min_overlap]
        r["symbol"] = "; ".join(s["symbol"] for s in hits)
        r["symbol_relation"] = "; ".join(s["relation"] for s in hits)
        r["symbol_line"] = "; ".join(str(s["line"]) for s in hits)
        linked += bool(hits)

    with Path(out or path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + [c for c in LINK_COLUMNS if c not in fields])
        w.writeheader()
        w.writerows(rows)
    return linked