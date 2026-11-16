"""One-page SETS pipeline: upload page images, detect symbols, build the graph,
trace back - and find similar words in the background.

Phase 1 (while the user waits, fast):
  1. save    images are kept in uploads/<run_id>/images/, so traces stay valid later
  2. detect  batch.run_batch: braces, slashes, ... -> results/symbols.csv
  3. clean   dataset_preprocess.clean: only the graph's columns -> results/symbols_clean.csv
  4. graph   set_trace.build_graph; the navigator is usable straight away

Phase 2 (background process, spot_worker.py, all CPU cores):
  5. words   similar words within the page (one image) or within and across
             pages (two or more); when done, the clusters are added to the
             graph as entities "Word 1", "Word 2", ...

The page checks the background process every few seconds and rebuilds the
graph once it finishes. Earlier runs can be reopened from the sidebar.

Start with:  streamlit run pipeline_app.py   (or press Run in PyCharm)
Project paths are set in spot_worker.py (CONFIG).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spot_worker import (MAX_MONTAGE_CLUSTERS, PROJECT_DIR, UPLOAD_ROOT,  # noqa: E402  (also sets sys.path)
                         read_status, results_dir, words_dir, write_status)
from batch import IMAGE_TYPES, run_batch  # noqa: E402
from dataset_preprocess import SCHEMAS, clean  # noqa: E402
from set_trace import build_graph, entities, find_entity, get_name  # noqa: E402
import trace_app as nav  # noqa: E402  (its main() is not run on import)

WORKER = Path(__file__).resolve().parent / "spot_worker.py"
POLL = "5s"                                   # how often the page checks the background job


# --------------------------------------------------------------------------- #
# Run folders
# --------------------------------------------------------------------------- #
def new_run() -> Path:
    """A new, kept folder for one upload: uploads/<date_time>/images/."""
    run = UPLOAD_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    (run / "images").mkdir(parents=True)
    return run


def symbols_path(run: Path) -> Path:
    return results_dir(run) / "symbols_clean.csv"


# --------------------------------------------------------------------------- #
# Phase 1: save, detect, clean, graph
# --------------------------------------------------------------------------- #
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


def save_user_annotations(run: Path, upload) -> None:
    """The uploaded annotations, cleaned, as annotations_user.csv."""
    if upload is None:
        return
    raw = run / "annotations_raw.csv"
    raw.write_bytes(upload.getbuffer())
    clean_csv(raw, run / "annotations_user.csv", "annotator")


def combine_annotations(run: Path) -> None:
    """annotations.csv = the user's annotations + the word clusters (once
    available). Header only if there are neither."""
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False)
              for f in (run / "annotations_user.csv", words_dir(run) / "word_annotations.csv")
              if f.exists()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.reindex(columns=SCHEMAS["annotator"]).to_csv(run / "annotations.csv", index=False)


def load_run(run: Path) -> None:
    """Build the graph of a run (with its words, if they are ready) and make it current."""
    combine_annotations(run)
    G = build_graph(str(run / "annotations.csv"), str(symbols_path(run)),
                    image_dir=str(run / "images"))
    st.session_state.update(run=run, G=G, focus=None, last_query=None,
                            words_in_graph=words_state(run).get("state") == "done")


# --------------------------------------------------------------------------- #
# Phase 2: similar words in a background process
# --------------------------------------------------------------------------- #
def start_words(run: Path) -> None:
    """Start spot_worker.py for this run. It is a separate process, so it keeps
    running across page reruns and may use parallel workers."""
    out = words_dir(run)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "log.txt").open("w") as log:
        proc = subprocess.Popen([sys.executable, str(WORKER), str(run)],
                                stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(PROJECT_DIR), start_new_session=True)
    write_status(run, state="running", pid=proc.pid, started=time.time())


def _alive(pid: int) -> bool:
    """True if the process is still running (a finished child is reaped first)."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)    # reap it if it was our child and has exited
        if done:
            return False
    except ChildProcessError:                     # not our child (e.g. after an app restart)
        pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def words_state(run: Path) -> dict:
    """status.json of the run; a "running" job whose process has gone is "failed"."""
    s = read_status(run)
    if s.get("state") == "running" and not _alive(int(s.get("pid", 0)) or -1):
        s = {**s, "state": "failed", "error": "the background process stopped unexpectedly"}
        write_status(run, **s)
    return s


def elapsed(s: dict) -> str:
    sec = int(s.get("seconds") or time.time() - s.get("started", time.time()))
    return f"{sec // 60} min {sec % 60} s" if sec >= 60 else f"{sec} s"


def log_tail(run: Path, lines: int = 15) -> str:
    path = words_dir(run) / "log.txt"
    return "\n".join(path.read_text().splitlines()[-lines:]) if path.exists() else ""


def words_banner(run: Path) -> None:
    """Status of the background job; adds the words to the graph when it finishes.
    Run as a fragment that refreshes itself while the job is running."""
    s = words_state(run)
    state = s.get("state")
    if state == "done" and not st.session_state.get("words_in_graph"):
        load_run(run)                             # rebuild the graph with the Word entities
        st.toast("Similar words added to the graph")
        st.rerun()                                # full page, so the graph and tabs update
    if state == "running":
        st.info(f"Finding similar words in the background ({elapsed(s)} so far). "
                "You can use the graph meanwhile; the Word entities are added when it finishes.")
    elif state == "failed":
        st.warning(f"Finding similar words failed: {s.get('error')}")
        if st.button("Retry similar words"):
            start_words(run)
            st.rerun()


def run_pipeline(images, annotations, spot_words: bool) -> None:
    """Phase 1 for one upload, with progress shown step by step; then phase 2
    is started in the background."""
    run = new_run()
    log = io.StringIO()
    with st.status(f"Processing {len(images)} image(s) ...", expanded=True) as status:
        try:
            st.write("Saving the images")
            save_images(images, run / "images")

            st.write("Detecting symbols")
            with contextlib.redirect_stdout(log):
                run_batch(folder=run / "images", words=False, cross="none")
            raw = results_dir(run) / "symbols.csv"
            if not raw.exists():
                raise RuntimeError("No symbols were produced; see the log for the failed images.")

            st.write("Cleaning the CSV")
            clean_csv(raw, symbols_path(run), "symbol")
            save_user_annotations(run, annotations)

            st.write("Building the graph")
            if spot_words:
                start_words(run)                  # phase 2 runs while the user explores
            load_run(run)
            label = (f"Symbols done: run {run.name}. Similar words are running in the background."
                     if spot_words else f"Done: run {run.name}")
            status.update(label=label, state="complete", expanded=False)
        except (Exception, SystemExit) as exc:   # run_batch reports bad input with SystemExit
            status.update(label="Processing failed", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
        finally:
            (run / "log.txt").write_text(log.getvalue())


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def natural_key(text: str) -> list:
    """Sort "Word 2" before "Word 10"."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def navigator(G) -> None:
    """The controls of trace_app.main, on the graph of the current run."""
    nav.G = G                                     # trace_app's helpers read these globals
    nav.ENTITY_SET = set(entities(G))
    state = st.session_state

    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    view = c1.radio("View", ["Focus", "Whole graph"], horizontal=True)
    depth = c2.number_input("Depth", 1, 6, 2, disabled=view != "Focus")
    query = c3.text_input("Jump to name or node id", placeholder="Brace, Word 3, page_1761 or _set6")
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
        labels = {e: get_name(G, e) or e for e in nav.ENTITY_SET}
        pick = st.selectbox("Start from", sorted(labels, key=lambda e: natural_key(labels[e])),
                            format_func=labels.get, index=None, placeholder="Choose an entity")
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


def words_view(run: Path) -> None:
    """The similar words of the run: counts, marked pages, cluster crops, table."""
    s = words_state(run)
    if s.get("state") == "running":
        st.info(f"Still running ({elapsed(s)}). Latest output:")
        st.code(log_tail(run) or "(no output yet)")
        return
    out = words_dir(run)
    if s.get("state") != "done":
        st.info("No similar words for this run (switched off, or failed: see the message above).")
        return

    view = st.radio("Show", ["Matches", "Clusters"], horizontal=True,
                    help="Matches: every accepted pair, before clustering. "
                         "Clusters: the groups built from the stronger matches; "
                         "only these become “Word N” in the graph.")
    if view == "Matches":
        matches_view(run)
    else:
        clusters_view(run, s)


def matches_view(run: Path) -> None:
    """Accepted matches within each page (within_<page>.jpg/.csv), strongest first."""
    out = words_dir(run)
    pages = sorted((p.stem.removeprefix("within_") for p in out.glob("within_*.jpg")),
                   key=natural_key)
    if not pages:
        st.info("No match images for this run.")
        return
    page = st.selectbox("Page", pages, key="matches_page")
    csv_path = out / f"within_{page}.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    c1, c2 = st.columns(2)
    c1.metric("Matches on this page", len(df))
    if not df.empty:
        c2.metric("Strong enough to cluster", int((df["z_context"] >= cluster_min_z()).sum()),
                  help=f"z ≥ {cluster_min_z()} (cluster_min_z)")
    st.caption("Left and right crops are one match; z is its score, ctx the bonus from "
               "neighbouring words. Matches between pages are not saved pair by pair; "
               "they appear in the Clusters view.")
    st.image(str(out / f"within_{page}.jpg"))
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download matches CSV", csv_path.read_bytes(),
                           file_name=f"matches_{page}_{run.name}.csv", mime="text/csv")


def clusters_view(run: Path, s: dict) -> None:
    """Word clusters: counts, marked pages, cluster crops, table."""
    out = words_dir(run)
    csv_path = out / "clusters.csv"
    df = pd.read_csv(csv_path, dtype={"page": str}) if csv_path.exists() else pd.DataFrame()
    if df.empty:
        st.info(f"No clusters: no match reached z ≥ {cluster_min_z()} (cluster_min_z). "
                "See the Matches view for the weaker matches.")
        return

    n_pages = df.groupby("cluster")["page"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Word clusters", len(n_pages))
    c2.metric("Within one page", int((n_pages == 1).sum()))
    c3.metric("Across pages", int((n_pages > 1).sum()))
    c4.metric("Time", elapsed(s))
    st.caption("Cluster N here is the entity “Word N” in the graph.")

    pages = sorted((p.stem.removeprefix("clusters_") for p in out.glob("clusters_*.jpg")
                    if p.stem != "clusters_overlay"), key=natural_key)
    page = st.selectbox("Page", pages, key="clusters_page")
    if page:
        st.image(str(out / f"clusters_{page}.jpg"))
    if len(pages) > 1 and (out / "clusters_overlay.jpg").exists():
        with st.expander("All pages together"):
            st.image(str(out / "clusters_overlay.jpg"))
    if (out / "clusters.jpg").exists():
        with st.expander(f"Cluster crops (largest {MAX_MONTAGE_CLUSTERS})"):
            st.image(str(out / "clusters.jpg"))

    st.dataframe(df, use_container_width=True)
    st.download_button("Download clusters CSV", csv_path.read_bytes(),
                       file_name=f"clusters_{run.name}.csv", mime="text/csv")


def cluster_min_z() -> float:
    """The cluster_min_z the worker uses (SPOT_PARAMS in spot_worker.py, else the default)."""
    from core.comparing_all_similar_words.batch_processing import CrossParams
    import spot_worker
    params = getattr(spot_worker, "SPOT_PARAMS", {})
    z = params.get("cluster_min_z", CrossParams.cluster_min_z)
    return float("-inf") if z is None else z


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
            for title, path in (("Symbols log", run / "log.txt"),
                                ("Similar words log", words_dir(run) / "log.txt")):
                if path.exists():
                    with st.expander(title):
                        st.text(path.read_text())

    with st.expander("1. Upload", expanded="G" not in st.session_state):
        images = st.file_uploader("Page images", accept_multiple_files=True,
                                  type=sorted(t.lstrip(".") for t in IMAGE_TYPES))
        annotations = st.file_uploader("Annotations CSV (optional)", type=["csv"],
                                       help="Columns: concept, name, page, x0, y0, x1, y1. "
                                            "page must match the image file name without extension.")
        spot_words = st.checkbox("Find similar words (in the background)", value=True,
                                 help="One image: within the page. Two or more: within each "
                                      "page and across pages. The graph is ready before this finishes.")
        if st.button("Process", type="primary", disabled=not images):
            run_pipeline(images, annotations, spot_words)

    if "G" in st.session_state:
        run = st.session_state.run
        running = words_state(run).get("state") == "running"
        st.fragment(run_every=POLL if running else None)(words_banner)(run)
        graph_tab, words_tab = st.tabs(["Graph and trace back", "Similar words"])
        with graph_tab:
            navigator(st.session_state.G)
        with words_tab:
            words_view(run)


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