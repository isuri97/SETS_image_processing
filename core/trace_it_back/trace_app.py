"""
Navigate the full SETS graph and trace any node back to the original images.

- The graph is drawn top-down like the Graphviz export: literals ($Anna, $x0,
  $40 …) at the top, the sets they belong to below.
- Click any node to focus on it: the view shows that node and its neighbours
  up to the chosen depth, and the right panel shows the image regions the
  node leads to.
- "Whole graph" shows everything; "Focus" keeps the view readable.

Press Run in PyCharm: the script relaunches itself under Streamlit.

Requires: pip install streamlit plotly pillow networkx pandas
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from set_trace import (ANNOTATIONS_CSV, INSTANCE, NAME, SYMBOLS_CSV,
                       box_of, build_graph, entities, entity_of, find_entity,
                       get_attr, get_name, image_of, is_instance_of, literal,
                       regions_on_image, trace)

CROP_MARGIN = 150            # pixels of context around each box
MAX_ARROWS = 400             # draw arrowheads only below this many edges
NODE_COLOURS = {"literal": "#F2CF5B", "special": "#9C9C9C", "entity": "#4C78A8",
                "image": "#59A14F", "region": "#E15759", "set": "#D9D9D9"}
RELATION_COLOURS = {"member": "blue", "note": "green"}


# --------------------------------------------------------------------------- #
# Node classification (for colour and for resolving a click to regions)
# --------------------------------------------------------------------------- #
def node_kind(G: nx.DiGraph, n: str) -> str:
    if n.startswith("$"):
        return "literal"
    if n in (INSTANCE, NAME):
        return "special"
    if is_instance_of(G, n, "image"):
        return "image"
    if is_instance_of(G, n, "image_region"):
        return "region"
    if n in ENTITY_SET:
        return "entity"
    return "set"


def resolve(G: nx.DiGraph, n: str) -> tuple[str, list[dict]]:
    """Map any clicked node to a title and the image regions it leads to."""
    kind = node_kind(G, n)
    if kind == "entity":
        return f"{get_name(G, n)} (entity {n})", trace(G, n)
    if kind == "region":
        owner = entity_of(G, n)
        regions = [r for r in trace(G, owner) if r["occurrence"] == n] if owner else []
        if not regions:                       # a member/note region
            regions = [region_record(G, n)]
        return f"Region {n}", regions
    if kind == "image":
        page = get_name(G, n)
        regions = []
        for r in regions_on_image(G, page):
            regions += [region_record(G, r["region"])]
        return f"Image {page}", regions
    if kind == "literal":                     # e.g. $Anna -> the entity it names
        hits = [e for e in find_entity(G, literal(n))
                if e in ENTITY_SET or is_instance_of(G, e, "image")]
        if hits:
            return resolve(G, hits[0])
        return f"Literal {n}", []
    # attribute / name / relation set: follow its members
    for m in G.predecessors(n):
        if node_kind(G, m) in ("region", "entity"):
            return resolve(G, m)
    return f"Set {n}", []


def region_record(G: nx.DiGraph, region: str) -> dict:
    img = image_of(G, region)
    owner = entity_of(G, region)
    if owner:
        hit = [r for r in trace(G, owner) if r["occurrence"] == region]
        if hit:
            return hit[0]
    return {"occurrence": region, "page": get_name(G, img),
            "source_file": literal(get_attr(G, img, "source_file")),
            "symbol_id": None, **box_of(G, region), "related": []}


# --------------------------------------------------------------------------- #
# Layout: top-down layers, like Graphviz "dot"
# --------------------------------------------------------------------------- #
def layered_layout(H: nx.DiGraph) -> dict[str, tuple[float, float]]:
    """Place nodes in layers by longest path from the sources, then order
    each layer by the mean x of its predecessors to reduce crossings."""
    layers: list[list[str]] = [list(g) for g in nx.topological_generations(H)]
    pos: dict[str, tuple[float, float]] = {}
    for depth, layer in enumerate(layers):
        if depth:
            layer.sort(key=lambda n: sum(pos[p][0] for p in H.predecessors(n) if p in pos)
                       / max(1, sum(1 for p in H.predecessors(n) if p in pos)))
        else:
            layer.sort()
        width = len(layer)
        for i, n in enumerate(layer):
            pos[n] = ((i - (width - 1) / 2) * 1.0, -depth * 1.5)
    return pos


def node_label(n: str) -> str:
    return n if len(n) <= 28 else n[:25] + "…"


def graph_figure(G: nx.DiGraph, H: nx.DiGraph, focus: str | None) -> go.Figure:
    pos = layered_layout(H)

    edge_x, edge_y = [], []
    for a, b in H.edges:
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                               line=dict(color="#BBBBBB", width=1),
                               hoverinfo="skip", showlegend=False))
    if H.number_of_edges() <= MAX_ARROWS:
        for a, b in H.edges:
            fig.add_annotation(x=pos[b][0], y=pos[b][1], ax=pos[a][0], ay=pos[a][1],
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=2, arrowsize=1,
                               arrowwidth=1, arrowcolor="#999999",
                               standoff=10, text="")

    for kind, colour in NODE_COLOURS.items():
        ids = [n for n in H if node_kind(G, n) == kind]
        if not ids:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids], y=[pos[n][1] for n in ids],
            mode="markers+text", name=kind, customdata=ids,
            text=[node_label(n) for n in ids], textposition="top center",
            textfont=dict(size=10),
            hovertemplate="%{customdata}<extra>" + kind + "</extra>",
            marker=dict(size=[22 if n == focus else 13 for n in ids], color=colour,
                        line=dict(color=["black" if n == focus else "white"
                                         for n in ids], width=2)),
        ))

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      clickmode="event+select", dragmode="pan",
                      legend=dict(orientation="h", y=-0.02))
    return fig


def neighbourhood(G: nx.DiGraph, focus: str, depth: int) -> nx.DiGraph:
    """Focus node plus everything within `depth` steps, either direction.
    The hub sets _set_instance and _set_name are shown but not expanded,
    otherwise every node would be two steps from every other."""
    keep, frontier = {focus}, {focus}
    for _ in range(depth):
        nxt = set()
        for n in frontier:
            if n in (INSTANCE, NAME) and n != focus:
                continue
            nxt |= set(G.predecessors(n)) | set(G.successors(n))
        frontier = nxt - keep
        keep |= nxt
    return G.subgraph(keep).copy()


def clicked_node(event) -> str | None:
    try:
        points = event.selection.points
    except AttributeError:
        return None
    if not points:
        return None
    data = points[0].get("customdata")
    return data[0] if isinstance(data, list) else data


# --------------------------------------------------------------------------- #
# Image panel
# --------------------------------------------------------------------------- #
def crop_region(r: dict) -> Image.Image | None:
    """Draw the region (red) and any linked member/note regions, then crop
    to the area covering all of them."""
    path = Path(r["source_file"])
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    boxes = [r] + r["related"]
    for rel in r["related"]:
        draw.rectangle((rel["x0"], rel["y0"], rel["x1"], rel["y1"]),
                       outline=RELATION_COLOURS.get(rel["relation"], "orange"),
                       width=3)
    draw.rectangle((r["x0"], r["y0"], r["x1"], r["y1"]), outline="red", width=4)
    crop = (max(0, min(b["x0"] for b in boxes) - CROP_MARGIN),
            max(0, min(b["y0"] for b in boxes) - CROP_MARGIN),
            min(img.width, max(b["x1"] for b in boxes) + CROP_MARGIN),
            min(img.height, max(b["y1"] for b in boxes) + CROP_MARGIN))
    return img.crop(crop)


def show_regions(title: str, regions: list[dict]) -> None:
    st.subheader(title)
    if not regions:
        st.write("This node does not lead to an image region.")
        return
    st.caption(f"{len(regions)} region(s)")
    for r in regions:
        label = f" {r['symbol_id']}" if r.get("symbol_id") else ""
        st.markdown(f"**{r['page']}{label}** · box ({r['x0']}, {r['y0']}, "
                    f"{r['x1']}, {r['y1']}) · `{r['source_file']}`")
        if r["related"]:
            st.caption(" · ".join(
                f"{rel['relation']} ({RELATION_COLOURS.get(rel['relation'], 'orange')}): "
                f"({rel['x0']}, {rel['y0']}, {rel['x1']}, {rel['y1']})"
                for rel in r["related"]))
        crop = crop_region(r)
        if crop is None:
            st.warning("Image file not found at this path.")
        else:
            st.image(crop)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_graph():
    return build_graph(ANNOTATIONS_CSV, SYMBOLS_CSV)


G = None
ENTITY_SET: set[str] = set()


def main() -> None:
    global G, ENTITY_SET
    st.set_page_config(page_title="SETS navigator", layout="wide")
    st.title("SETS graph navigator")
    G = load_graph()
    ENTITY_SET = set(entities(G))
    state = st.session_state
    state.setdefault("focus", None)

    # Controls
    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    view = c1.radio("View", ["Focus", "Whole graph"], horizontal=True)
    depth = c2.number_input("Depth", 1, 6, 2, disabled=view != "Focus")
    name = c3.text_input("Jump to name or node id", placeholder="Ambel or _set6")
    if c4.button("Reset"):
        state.focus = None
    if name and name != state.get("last_query"):   # apply a new search once
        state.last_query = name
        query = name.strip()
        hits = [query] if query in G else find_entity(G, query.capitalize())
        if hits:
            state.focus = hits[0]
        else:
            st.info(f"Nothing called {name!r}.")

    if view == "Focus" and state.focus is None:
        st.write("Pick a starting entity:")
        cols = st.columns(max(1, len(ENTITY_SET)))
        for col, e in zip(cols, sorted(ENTITY_SET, key=lambda x: get_name(G, x))):
            if col.button(get_name(G, e), key=f"start_{e}"):
                state.focus = e
                st.rerun()
        return

    H = neighbourhood(G, state.focus, depth) if view == "Focus" else G
    left, right = st.columns([3, 2])
    with left:
        st.caption(f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges · "
                   "click a node to focus on it · drag to pan, scroll to zoom")
        event = st.plotly_chart(graph_figure(G, H, state.focus),
                                on_select="rerun", selection_mode="points",
                                key=f"graph_{view}_{state.focus}_{depth}")
        clicked = clicked_node(event)
        if clicked and clicked != state.focus:
            state.focus = clicked
            st.rerun()

    with right:
        if state.focus:
            show_regions(*resolve(G, state.focus))
        else:
            st.write("Click a node.")


def running_in_streamlit() -> bool:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    if running_in_streamlit():
        main()
    else:   # launched from PyCharm's Run button
        from streamlit.web import cli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(cli.main())