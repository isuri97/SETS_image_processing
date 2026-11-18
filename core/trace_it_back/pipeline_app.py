"""One-page SETS pipeline: upload page images, detect symbols, build the graph, trace back.

Everything runs internally after an upload; the user never edits code:

  1. save    images are kept in uploads/<run_id>/images/, so traces stay valid later
  2. detect  batch.run_batch: braces, slashes, ... -> results/symbols.csv
  3. clean   dataset_preprocess.clean: only the graph's columns -> results/symbols_clean.csv
  4. graph   set_trace.build_graph, with source_file resolved against the images folder
  5. trace   the navigator of trace_app: click a node, see the image regions it leads to

Earlier runs can be reopened from the sidebar without processing again.

Start with:  streamlit run pipeline_app.py   (or press Run in PyCharm)
"""
from __future__ import annotations

import contextlib
import io
import sys
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from set_trace import build_graph, entities, find_entity, get_name, is_instance_of  # noqa: E402
# =========================================================================== #
# CONFIG - set once: the folders holding the modules, relative to the project root
# =========================================================================== #
PROJECT_DIR = next(p for p in Path(__file__).resolve().parents if (p / "core").is_dir())
MODULE_DIRS = [
    "core/marks",        # batch.py, run.py, symbol_csv.py
    "core/preprocess",   # dataset_preprocess.py
    "core/trace",        # set_trace.py, trace_app.py
]
UPLOAD_ROOT = PROJECT_DIR / "uploads"
# =========================================================================== #

for d in MODULE_DIRS:
    sys.path.insert(0, str(PROJECT_DIR / d))

from batch import IMAGE_TYPES, run_batch  # noqa: E402
from dataset_preprocess import SCHEMAS, clean  # noqa: E402
from set_trace import build_graph, entities, find_entity, get_name  # noqa: E402
import trace_app as nav  # noqa: E402  (its main() is not run on import)


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def new_run() -> Path:
    """A new, kept folder for one upload: uploads/<date_time>/images/."""
    run = UPLOAD_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run / "images").mkdir(parents=True)
    return run


def save_images(files, folder: Path) -> None:
    for f in files:
        (folder / Path(f.name).name).write_bytes(f.getbuffer())  # drop client-side paths


def clean_csv(src: Path, dst: Path, schema: str) -> Path:
    """Keep only the columns of `schema` (dataset_preprocess) and write them to dst."""
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    out, _, missing = clean(df, schema)
    if missing:
        raise ValueError(f"{src.name} lacks the columns {', '.join(missing)}")
    out.to_csv(dst, index=False)
    return dst


def annotations_for(run: Path, upload) -> Path:
    """The run's annotations.csv: the uploaded one, cleaned, or an empty one
    (header only) so the graph is built from the symbols alone."""
    dst = run / "annotations.csv"
    if upload is None:
        pd.DataFrame(columns=SCHEMAS["annotator"]).to_csv(dst, index=False)
        return dst
    raw = run / "annotations_raw.csv"
    raw.write_bytes(upload.getbuffer())
    return clean_csv(raw, dst, "annotator")


def symbols_path(run: Path) -> Path:
    return run / "images" / "results" / "symbols_clean.csv"


def load_run(run: Path) -> None:
    """Build the graph of a processed run and make it the current one."""
    G = build_graph(str(run / "annotations.csv"), str(symbols_path(run)),
                    image_dir=str(run / "images"))
    st.session_state.update(run=run, G=G, focus=None, last_query=None)


def run_pipeline(images, annotations) -> None:
    """Steps 1-4 for one upload, with progress shown step by step."""
    run = new_run()
    log = io.StringIO()
    with st.status(f"Processing {len(images)} image(s) ...", expanded=True) as status:
        try:
            st.write("Saving the images")
            save_images(images, run / "images")

            st.write("Detecting symbols")
            with contextlib.redirect_stdout(log):
                run_batch(folder=run / "images", words=False, cross="none")
            raw = run / "images" / "results" / "symbols.csv"
            if not raw.exists():
                raise RuntimeError("No symbols were produced; see the log for the failed images.")

            st.write("Cleaning the CSV")
            clean_csv(raw, symbols_path(run), "symbol")
            annotations_for(run, annotations)

            st.write("Building the graph")
            load_run(run)
            status.update(label=f"Done: run {run.name}", state="complete", expanded=False)
        except (Exception, SystemExit) as exc:  # run_batch reports bad input with SystemExit
            status.update(label="Processing failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
        finally:
            (run / "log.txt").write_text(log.getvalue())


# --------------------------------------------------------------------------- #
# Navigator (the controls of trace_app.main, on the graph of the current run)
# --------------------------------------------------------------------------- #
def navigator(G) -> None:
    nav.G = G                                  # trace_app's helpers read these globals
    nav.ENTITY_SET = set(entities(G))
    state = st.session_state

    def natural_key(text: str) -> list:
        """Sort "Word 2" before "Word 10"."""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]

    def entity_label(G, e: str) -> str:
        """Dropdown label: the entity's name and how many image regions it has, e.g. "Brace (12)"."""
        n = sum(is_instance_of(G, s, "image_region") for s in G.successors(e))
        return f"{get_name(G, e) or e} ({n})"


    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    view = c1.radio("View", ["Focus", "Whole graph"], horizontal=True)
    depth = c2.number_input("Depth", 1, 6, 2, disabled=view != "Focus")
    query = c3.text_input("Jump to name or node id", placeholder="Brace, page_1761 or _set6")
    if c4.button("Reset"):
        state.focus = None
    if query and query != state.get("last_query"):   # apply a new search once
        state.last_query = query
        q = query.strip()
        hits = [q] if q in G else find_entity(G, q) or find_entity(G, q.capitalize())
        if hits:
            state.focus = hits[0]
        else:
            st.info(f"Nothing called {query!r}.")

    if view == "Focus" and state.focus is None:
        labels = {e: entity_label(G, e) for e in nav.ENTITY_SET}
        pick = st.selectbox("Start from", sorted(labels, key=lambda e: natural_key(labels[e])),
                            format_func=labels.get, index=None,
                            placeholder="Choose an entity (type to search)")
        if pick:
            state.focus = pick
            st.rerun()
        return

    H = nav.neighbourhood(G, state.focus, depth) if view == "Focus" else G
    left, right = st.columns([3, 2])
    with left:
        st.caption(f"{H.number_of_nodes()} nodes, {H.number_of_edges()} edges · "
                   "click a node to focus on it · drag to pan, scroll to zoom")
        event = st.plotly_chart(nav.graph_figure(G, H, state.focus),
                                on_select="rerun", selection_mode="points",
                                key=f"graph_{state.run.name}_{view}_{state.focus}_{depth}")
        clicked = nav.clicked_node(event)
        if clicked and clicked != state.focus:
            state.focus = clicked
            st.rerun()
    with right:
        if state.focus:
            nav.show_regions(*nav.resolve(G, state.focus))
        else:
            st.write("Click a node.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="SETS pipeline", layout="wide")
    st.title("SETS: upload, detect, trace")

    with st.sidebar:
        st.header("Earlier runs")
        runs = sorted((p for p in UPLOAD_ROOT.glob("*") if symbols_path(p).exists()),
                      reverse=True) if UPLOAD_ROOT.exists() else []
        choice = st.selectbox("Run", [r.name for r in runs], index=None,
                              placeholder="Choose a run")
        if st.button("Open", disabled=choice is None):
            load_run(UPLOAD_ROOT / choice)
        if "run" in st.session_state:
            run = st.session_state.run
            st.caption(f"Current run: {run.name}")
            st.download_button("Download symbols CSV", symbols_path(run).read_bytes(),
                               file_name=f"symbols_{run.name}.csv", mime="text/csv")

    with st.expander("1. Upload", expanded="G" not in st.session_state):
        images = st.file_uploader("Page images", accept_multiple_files=True,
                                  type=sorted(t.lstrip(".") for t in IMAGE_TYPES))
        annotations = st.file_uploader("Annotations CSV (optional)", type=["csv"],
                                       help="Columns: concept, name, page, x0, y0, x1, y1. "
                                            "page must match the image file name without extension.")
        if st.button("Process", type="primary", disabled=not images):
            run_pipeline(images, annotations)

    if "G" in st.session_state:
        st.subheader("2. Graph and trace back")
        navigator(st.session_state.G)


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