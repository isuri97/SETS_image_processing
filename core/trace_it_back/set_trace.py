"""
Build a SETS-style graph from annotations.csv and symbols.csv, and trace any
entity back to the image regions it came from.

Graph conventions (same as the existing SETS graph)
---------------------------------------------------
- Literal nodes are labelled "$value" (e.g. "$Anna", "$x0", "$40").
- Set nodes are "_setN". An edge  member -> set  means "member is in set".
- x is an instance of type T      :  _set_instance -> x,  T -> x
- x is named "N"                  :  {_set_name, x, $N}
- x has attribute key = value     :  {x, $key, $value}

The one structural change from the current graph
------------------------------------------------
Each occurrence gets an explicit relation to its image entity:
    {occurrence, image, $on_image}
so tracing no longer depends on the $page value and the image name sharing
the same literal node.

Run this file directly to print every trace (no Streamlit needed).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

# =========================================================================== #
# CONFIG
# =========================================================================== #
ANNOTATIONS_CSV = "annotations.csv"
SYMBOLS_CSV = "symbols1.csv"
# Fallback folder for pages with no source_file in symbols.csv: <IMAGE_DIR>/<page>.png
IMAGE_DIR = "data/image"
BOX_KEYS = ("x0", "y0", "x1", "y1")
# =========================================================================== #

INSTANCE, NAME = "_set_instance", "_set_name"


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
class SetsBuilder:
    """Minimal helpers that write SETS patterns into a NetworkX DiGraph."""

    def __init__(self) -> None:
        self.G = nx.DiGraph()
        self.G.add_nodes_from([INSTANCE, NAME])
        self._next = 0

    def lit(self, value) -> str:
        node = f"${value}"
        self.G.add_node(node)
        return node

    def new_set(self, *members: str) -> str:
        node = f"_set{self._next}"
        self._next += 1
        self.G.add_node(node)
        for m in members:
            self.G.add_edge(m, node)
        return node

    def instance(self, type_name: str) -> str:
        x = self.new_set()
        self.G.add_edge(INSTANCE, x)
        self.G.add_edge(self.lit(type_name), x)
        return x

    def name(self, x: str, value) -> None:
        self.new_set(NAME, x, self.lit(value))

    def attr(self, x: str, key: str, value) -> None:
        self.new_set(x, self.lit(key), self.lit(value))

    def relate(self, a: str, b: str, relation: str) -> None:
        self.new_set(a, b, self.lit(relation))


def read_table(path: str, required: list[str]) -> pd.DataFrame:
    """Read a CSV relative to this script, tolerating BOMs, spaces and ';'."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    df = pd.read_csv(p, dtype=str, sep=None, engine="python",
                     encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p} is missing columns {missing}; "
                         f"found {list(df.columns)}")
    return df

def resolve_image(name: str, image_dir: str) -> str:
    """Full path of a photo: absolute paths are kept, file names are joined to image_dir."""
    path = Path(name)
    if path.is_absolute():
        return str(path)                        # older CSVs with full paths still work
    folder = Path(image_dir)
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    return str(folder / path)

def build_graph(annotations_csv: str, symbols_csv: str,
                image_dir: str = IMAGE_DIR) -> nx.DiGraph:
    ann = read_table(annotations_csv, ["concept", "name", "page", *BOX_KEYS])
    sym = read_table(symbols_csv, ["page", "kind", "ox0", "oy0", "ox1", "oy1"])
    b = SetsBuilder()

    # 1. One image entity per page, carrying its file path.
    #    source_file is optional: pages without it fall back to IMAGE_DIR.
    paths = (dict(zip(sym["page"], sym["source_file"]))
             if "source_file" in sym.columns else {})
    images: dict[str, str] = {}
    for page in sorted(set(ann["page"]) | set(sym["page"])):
        img = b.instance("image")
        b.name(img, page)
        b.attr(img, "source_file", resolve_image(paths.get(page) or f"{page}.png", image_dir))
        images[page] = img

    # 2. One entity per (concept, name); one region per annotated box.
    entities: dict[tuple[str, str], str] = {}

    def add_region(page: str, box: list) -> str:
        region = b.instance("image_region")
        b.relate(region, images[page], "on_image")      # explicit link to image
        for k, v in zip(BOX_KEYS, box):
            b.attr(region, k, v)
        return region

    def add_occurrence(type_name: str, name: str, page: str, box: list) -> str:
        key = (type_name, name)
        if key not in entities:
            entities[key] = b.instance(type_name)
            b.name(entities[key], name)
        occ = add_region(page, box)
        b.G.add_edge(entities[key], occ)                # occurrence of entity
        return occ

    for r in ann.itertuples():
        add_occurrence(r.concept, r.name.capitalize(), r.page,
                       [r.x0, r.y0, r.x1, r.y1])

    # 3. Symbols. A row with relation == "symbol" is the mark itself; other
    #    rows (member, note, ...) are regions linked to that mark by their
    #    relation, e.g. {brace_B1, region, $member}.
    if "relation" not in sym.columns:
        sym["relation"] = "symbol"
    sym_id = (sym["page"] + ":" + sym["symbol"]) if "symbol" in sym.columns \
        else sym.index.astype(str)
    marks: dict[str, str] = {}
    for idx in sym.index[sym["relation"] == "symbol"]:
        r = sym.loc[idx]
        occ = add_occurrence("symbol", r["kind"].capitalize(), r["page"],
                             [r.ox0, r.oy0, r.ox1, r.oy1])
        b.attr(occ, "symbol_id", r.get("symbol", idx))
        marks[sym_id[idx]] = occ
    for idx in sym.index[sym["relation"] != "symbol"]:
        r = sym.loc[idx]
        mark = marks.get(sym_id[idx])
        if mark is None:
            print(f"[warn] {sym_id[idx]} {r['relation']}: no symbol row, skipped")
            continue
        region = add_region(r["page"], [r.ox0, r.oy0, r.ox1, r.oy1])
        b.relate(mark, region, r["relation"])

    return b.G


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def literal(node: str) -> str:
    return node[1:] if node.startswith("$") else node


def is_instance_of(G: nx.DiGraph, x: str, type_name: str) -> bool:
    preds = set(G.predecessors(x))
    return INSTANCE in preds and f"${type_name}" in preds


def get_attr(G: nx.DiGraph, x: str, key: str) -> str | None:
    """Value of attribute `key` on x, i.e. the third member of {x, $key, $v}."""
    for s in G.successors(x):
        preds = set(G.predecessors(s))
        if f"${key}" in preds:
            rest = preds - {x, f"${key}"}
            if len(rest) == 1:
                return next(iter(rest))
    return None


def get_name(G: nx.DiGraph, x: str) -> str | None:
    return get_attr_like(G, x, NAME)


def get_attr_like(G: nx.DiGraph, x: str, marker: str) -> str | None:
    for s in G.successors(x):
        preds = set(G.predecessors(s))
        if marker in preds:
            rest = preds - {x, marker}
            return literal(next(iter(rest))) if rest else None
    return None


def find_entity(G: nx.DiGraph, name: str) -> list[str]:
    """All entities named `name` (e.g. 'Anna')."""
    lit = f"${name}"
    if lit not in G:
        return []
    out = []
    for s in G.successors(lit):
        if NAME in G.predecessors(s):
            out += [p for p in G.predecessors(s) if p not in (NAME, lit)]
    return out


def image_of(G: nx.DiGraph, occ: str) -> str | None:
    """Follow the explicit {occ, image, $on_image} relation."""
    return get_attr(G, occ, "on_image")


def box_of(G: nx.DiGraph, region: str) -> dict[str, int]:
    return {k: int(float(literal(get_attr(G, region, k)))) for k in BOX_KEYS}


def related_regions(G: nx.DiGraph, mark: str) -> list[dict]:
    """Regions linked to a symbol by {mark, region, $relation} (member, note…)."""
    out = []
    for s in G.successors(mark):
        preds = set(G.predecessors(s)) - {mark}
        region = next((p for p in preds if is_instance_of(G, p, "image_region")), None)
        rel = next((p for p in preds if p.startswith("$")), None)
        if region and rel and rel != "$on_image":
            out.append({"relation": literal(rel), **box_of(G, region)})
    return out


def trace(G: nx.DiGraph, entity: str) -> list[dict]:
    """Every image region of an entity: file path, page, box and, for
    symbols, the member/note regions linked to it."""
    regions = []
    for occ in G.successors(entity):
        if not is_instance_of(G, occ, "image_region"):
            continue
        img = image_of(G, occ)
        symbol_id = get_attr(G, occ, "symbol_id")
        regions.append({
            "occurrence": occ,
            "page": get_name(G, img),
            "source_file": literal(get_attr(G, img, "source_file")),
            "symbol_id": literal(symbol_id) if symbol_id else None,
            **box_of(G, occ),
            "related": related_regions(G, occ),
        })
    return regions


def entity_of(G: nx.DiGraph, region: str) -> str | None:
    """The entity a region is an occurrence of (None for member/note regions)."""
    return next((p for p in G.predecessors(region)
                 if p.startswith("_set") and p != INSTANCE), None)


def regions_on_image(G: nx.DiGraph, page: str) -> list[dict]:
    """Reverse direction: everything annotated on one image."""
    img = next((i for i in find_entity(G, page) if is_instance_of(G, i, "image")), None)
    if img is None:
        return []
    out = []
    for s in G.successors(img):
        if "$on_image" not in G.predecessors(s):
            continue
        region = next(p for p in G.predecessors(s) if p not in (img, "$on_image"))
        entity = entity_of(G, region)
        out.append({"region": region,
                    "entity": get_name(G, entity) if entity else "(linked region)",
                    **box_of(G, region)})
    return out


def entities(G: nx.DiGraph) -> list[str]:
    """All annotated entities (anything with image_region occurrences)."""
    return [n for n in G if n.startswith("_set") and n not in (INSTANCE, NAME) and any(
        is_instance_of(G, s, "image_region") for s in G.successors(n))]


if __name__ == "__main__":
    G = build_graph(ANNOTATIONS_CSV, SYMBOLS_CSV)
    print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    for e in entities(G):
        print(f"{get_name(G, e)} ({e}) ->")
        for r in trace(G, e):
            related = r.pop("related")
            print("   ", r)
            for rel in related:
                print("       ", rel)

    for page in sorted({get_name(G, i) for i in G if is_instance_of(G, i, "image")}):
        print(f"\nOn {page}:")
        for r in regions_on_image(G, page):
            print("   ", r)