"""
Streamlit app for reviewing and annotating word clusters (app.py).

Run from the project root:
    streamlit run core/comparing_all_similar_words/app.py

The app is a front end only: all data and actions live in annotation.py
(CSV files in the annotation folder), the "does this word fit?" check in
review.py, and the optional automatic names in recognition.py.

For each cluster the annotator can
  - see every word, with the check's verdict on each one:
        fits                 - looks like the rest of the cluster
        fits #N better       - looks more like cluster N (suggested move)
        fits no cluster      - suggested removal
  - compare with the suggested cluster side by side
  - move a word to another cluster or to a new one, or remove it
  - accept all suggestions of the cluster at once
  - assign the cluster to a concept of the list, or define a new concept;
    its value (prefilled with the automatic name) becomes a signifier of it
  - remove, restore or merge whole clusters
  - export the result (CSV or W3C Web Annotation JSON-LD)

Settings (optional environment variables):
    ANNOTATION_DIR   annotation folder  (default: result/annotations next to this file)
    RESULT_DIR       folder with clusters.csv, for the first import (default: result/)
    IMAGE_DIR        folder with the page images, for the first import
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:                    # sibling modules, however the app is started
    sys.path.insert(0, str(HERE))

import recognition                               # noqa: E402
import review
# noqa: E402
from annotation import AnnotationError, ClusterStore   # noqa: E402

# Full-width buttons and images: newer Streamlit uses width="stretch",
# older versions use_container_width=True (since removed in favour of width).
FULL = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
        else {"use_container_width": True})

VERDICT = {"keep": ("✅", "fits"), "single": ("➖", "only word"),
           "move": ("↪️", "fits #{n} better"), "remove": ("❌", "fits no cluster")}


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def get_store(folder: str) -> ClusterStore:
    """One store per session and folder (keeps the page-image cache). The
    files are re-read on every interaction, so changes made elsewhere (the
    command line, another tab) are seen and never overwritten."""
    key = f"store::{folder}"
    if key not in st.session_state:
        st.session_state[key] = ClusterStore(folder)
    else:
        st.session_state[key].reload()
    return st.session_state[key]


def flash(kind: str, text: str) -> None:
    """Message shown after the next rerun."""
    st.session_state["flash"] = (kind, text)


def show_flash() -> None:
    kind, text = st.session_state.pop("flash", (None, None))
    if kind == "ok":
        st.success(text)
    elif kind == "error":
        st.error(text)


def do(store: ClusterStore, run: str, done: str, fn, *args, **kwargs) -> None:
    """Button callback for one store action. Callbacks run before the page
    is redrawn, with the arguments the annotator saw when clicking. The
    files are re-read first, then the action runs; `done` is shown on
    success, otherwise the reason it was refused. Automatic names are
    refreshed, since they follow the clusters' current words."""
    store.reload()
    try:
        fn(*args, **kwargs)
    except AnnotationError as e:
        flash("error", str(e))
    else:
        flash("ok", done)
        if recognition.load_readings(store):
            recognition.auto_name_clusters(store, run)


def word_image(store: ClusterStore, run: str, occ, context: bool):
    return cv2.cvtColor(store.crop(occ, run, context=context), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def import_form(store: ClusterStore) -> None:
    st.info("No clusters imported yet.")
    with st.form("import"):
        result = st.text_input("Folder with clusters.csv",
                               os.environ.get("RESULT_DIR", str(HERE / "result")))
        images = st.text_input("Folder with the page images", os.environ.get("IMAGE_DIR", ""))
        if st.form_submit_button("Import"):
            try:
                run = store.import_run(result, image_dir=images or None)
                flash("ok", f"Imported run '{run}'")
            except (AnnotationError, FileNotFoundError) as e:
                flash("error", str(e))
            st.rerun()


def cluster_option(c: dict, flags: dict[int, int]) -> str:
    n = flags.get(c["cluster_id"], 0)
    flag = f" · ⚠ {n}" if n else ""
    return f"#{c['number']} · {c['status']} · {c['size']} words{flag} · {ClusterStore.label_text(c)}"


def sidebar(store: ClusterStore) -> tuple[str, dict]:
    """Settings, progress, cluster navigation and export. Returns the run
    and the check settings."""
    sb = st.sidebar
    sb.title("Word clusters")
    st.session_state.setdefault("annotator", os.environ.get("USER", ""))
    sb.text_input("Annotator", key="annotator")

    runs = store.runs()
    run = sb.selectbox("Run", runs, index=len(runs) - 1) if len(runs) > 1 else runs[0]

    p = store.progress(run)
    done = p["labelled"] + p["removed"] + p["merged"]
    sb.progress(done / max(p["total"], 1),
                text=f"{done}/{p['total']} decided · {p['concept']} concepts · "
                     f"{p['signifier']} signifiers · {p['removed']} removed")

    with sb.expander("Check settings"):
        thr = dict(min_z=st.slider("Fit threshold (min z)", 0.5, 3.0, 1.5, 0.1),
                   margin=st.slider("Better-cluster margin", 0.0, 2.0, 0.5, 0.1))
    context = sb.toggle("Show words in their line", value=False)
    sb.slider("Words per row", 2, 6, 3, key="per_row")
    show = sb.radio("Show clusters", ["pending", "all", "labelled", "removed", "with flags"],
                    horizontal=True)
    st.session_state["context"] = context

    checks = review.check(store, run, **thr)
    flags: dict[int, int] = {}
    for c in checks:
        if c.recommendation in ("move", "remove"):
            flags[c.cluster_id] = flags.get(c.cluster_id, 0) + 1

    clusters = [c for c in store.list_clusters(run) if c["status"] != "merged"]
    if show == "with flags":
        clusters = [c for c in clusters if flags.get(c["cluster_id"])]
    elif show != "all":
        clusters = [c for c in clusters if c["status"] == show]
    if not clusters:
        sb.warning("No clusters to show with this filter.")
        st.session_state["cid"] = None
        return run, dict(thr=thr, checks=checks)

    # The selector is driven through its session key: a changing `index`
    # would make Streamlit treat it as a new widget and lose the choice.
    ids = [c["cluster_id"] for c in clusters]
    if st.session_state.get("cluster_select") not in ids:
        st.session_state["cluster_select"] = ids[0]
    pos = ids.index(st.session_state["cluster_select"])

    def select(cid: int) -> None:
        st.session_state["cluster_select"] = cid

    prev_col, next_col = sb.columns(2)
    prev_col.button("◀ Previous", disabled=pos == 0, on_click=select,
                    args=(ids[max(pos - 1, 0)],), **FULL)
    next_col.button("Next ▶", disabled=pos == len(ids) - 1, on_click=select,
                    args=(ids[min(pos + 1, len(ids) - 1)],), **FULL)
    labels = {c["cluster_id"]: cluster_option(c, flags) for c in clusters}
    sb.selectbox("Cluster", ids, key="cluster_select", format_func=labels.get)
    st.session_state["cid"] = st.session_state["cluster_select"]

    with sb.expander("Export"):
        for fmt, suffix, mime in (("CSV", ".csv", "text/csv"),
                                  ("JSON-LD", ".jsonld", "application/ld+json")):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                tmp = Path(f.name)
            if fmt == "CSV":
                n = store.export_csv(tmp, run)
            else:
                n = store.export_web_annotations(tmp, run)
            st.download_button(f"{fmt} ({n} words)", tmp.read_bytes(), f"labels_{run}{suffix}",
                               mime, **FULL)
            tmp.unlink()

    with sb.expander("Automatic names (TrOCR)"):
        st.caption("Reads every word not read yet and suggests a name per cluster.")
        if st.button("Run TrOCR", **FULL):
            try:
                with st.spinner("Reading words…"):
                    res = recognition.auto_name_clusters(store, run, recognition.TrOCRRecogniser())
                flash("ok", f"{res['read']} words read; {res['named']} clusters named, "
                            f"{res['unagreed']} without agreement")
            except ImportError as e:
                flash("error", str(e))
            st.rerun()
    return run, dict(thr=thr, checks=checks)


# --------------------------------------------------------------------------- #
# Cluster view
# --------------------------------------------------------------------------- #
def verdict_text(chk) -> str:
    if chk is None:
        return ""
    icon, text = VERDICT[chk.recommendation]
    return f"{icon} {text.format(n=chk.best_number)}"


def target_options(store: ClusterStore, run: str, exclude: int) -> dict[int, str]:
    """Open clusters a word can move to, as short labels: '#1 Padova'
    (its name, else its automatic name, else nothing)."""
    return {c["cluster_id"]: f"#{c['number']} {c['name'] or c.get('auto_name') or ''}".strip()
            for c in store.list_clusters(run)
            if c["status"] in ("pending", "labelled") and c["cluster_id"] != exclude}


def word_tile(store: ClusterStore, run: str, cid: int, o, chk, readings: dict,
              targets: dict[int, str], annotator: str) -> None:
    """One word: image, check verdict, reading, and its actions."""
    context = st.session_state.get("context", False)
    with st.container(border=True):
        st.image(word_image(store, run, o, context), **FULL)
        st.caption(f"occ {o.occ_id} · {o.page}")
        line = verdict_text(chk)
        if o.occ_id in readings and readings[o.occ_id][0]:
            line += f" · read: *{readings[o.occ_id][0]}*"
        flagged = chk is not None and chk.recommendation in ("move", "remove")
        st.markdown(f"**{line}**" if flagged else line)

        # row 1: target cluster + Move; row 2: New cluster + Remove
        ids = list(targets)
        suggested = chk.best_cluster if chk is not None and chk.recommendation == "move" else None
        pick, move = st.columns([3, 2], vertical_alignment="bottom")
        target = pick.selectbox("Move to", ids, format_func=targets.get,
                                label_visibility="collapsed",
                                index=ids.index(suggested) if suggested in ids else 0,
                                key=f"to_{o.occ_id}_{suggested}") if ids else None
        move.button("Move", key=f"mv_{o.occ_id}", disabled=target is None,
                    type="primary" if suggested is not None else "secondary", **FULL,
                    on_click=do, args=(store, run, f"occ {o.occ_id} moved to "
                                       f"{targets.get(target, '').split(' ')[0]}",
                                       store.move_occurrence, o.occ_id, target, annotator))
        new, rm = st.columns(2)
        new.button("New cluster", key=f"new_{o.occ_id}", **FULL,
                   help="Move this word into a new cluster of its own",
                   on_click=do, args=(store, run, f"occ {o.occ_id} moved to a new cluster",
                                      store.new_cluster, [o.occ_id], annotator))
        rm.button("Remove", key=f"rm_{o.occ_id}", help="Remove this word (not in any cluster)",
                  type="primary" if chk is not None and chk.recommendation == "remove"
                  else "secondary", **FULL,
                  on_click=do, args=(store, run, f"occ {o.occ_id} removed",
                                     store.remove_occurrence, cid, o.occ_id, annotator))


EXISTING, NEW = "Existing concept", "New concept"


def save_label(store: ClusterStore, run: str, c: dict, annotator: str, keys: dict) -> None:
    """Callback of the Save button: assign the cluster to the chosen
    existing concept, or define the typed new concept first. The cluster's
    value becomes a signifier of that concept; the automatic value left
    unchanged is recorded as accepted (name_source auto), an edited or typed
    one as manual."""
    new = st.session_state[keys["mode"]] == NEW
    concept = (st.session_state.get(keys["new"] if new else keys["concept"]) or "").strip()
    value = st.session_state[keys["value"]].strip() or None
    note = st.session_state[keys["note"]].strip() or None
    auto = value is not None and value == c.get("auto_name") and \
        (c["name"] is None or c.get("name_source") == "auto")
    do(store, run, f"cluster #{c['number']}: '{value}' is now a signifier of "
                   f"{'the new ' if new else ''}concept '{concept}'",
       store.assign, c["cluster_id"], concept, annotator,
       value=None if auto else value, note=note, new=new)


def concept_group(store: ClusterStore, concept: str | None) -> None:
    """The signifiers a concept already has (over all runs), so the
    annotator can see whether this cluster belongs with them."""
    if not concept:
        return
    linked = store.concept_clusters(concept)
    st.caption(f"**{concept}** · " + (" · ".join(f"*{x['name'] or '?'}* #{x['number']}"
                                                   for x in linked)
                                        if linked else "no signifiers yet"))


def manage_concepts(store: ClusterStore, run: str, annotator: str) -> None:
    """The concept list with each concept's signifiers, and a Delete button
    per concept. Delete is only possible while no cluster is assigned to the
    concept (unassign those clusters first)."""
    concepts = store.concepts()
    with st.popover(f"Manage concepts ({len(concepts)})", **FULL):
        if not concepts:
            st.caption("The concept list is empty.")
            return
        for name in concepts:
            users = store.concept_users(name)
            text, button = st.columns([4, 1], vertical_alignment="center")
            sig = ", ".join(store.signifiers(name)) or "no signifiers"
            text.markdown(f"**{name}**  \n:gray[{sig}]")
            in_use = ", ".join(f"#{u['number']}" for u in users[:6]) + \
                ("…" if len(users) > 6 else "")
            button.button("🗑", key=f"delc_{name}", disabled=bool(users),
                          help=f"used by cluster(s) {in_use} - unassign them first"
                          if users else f"delete '{name}' from the list",
                          on_click=do, args=(store, run, f"concept '{name}' deleted",
                                             store.delete_concept, name, annotator))


def label_panel(store: ClusterStore, run: str, c: dict, annotator: str) -> None:
    """Concept assignment, signifier value, and whole-cluster actions."""
    cid = c["cluster_id"]
    if c["status"] == "removed":
        st.warning(f"This cluster was removed{': ' + c['note'] if c.get('note') else ''}.")
        st.button("Restore cluster", key=f"rsc_{cid}", on_click=do,
                  args=(store, run, f"cluster #{c['number']} restored", store.restore,
                        cid, annotator))
        return

    # widget keys include the stored values (and the size of the concept
    # list), so the box shows fresh values whenever one of them changes
    concepts = store.concepts()
    auto = c.get("auto_name")
    tag = f"{cid}_{c['name']}_{auto}_{c.get('concept')}_{len(concepts)}"
    keys = {k: f"{k}_{tag}" for k in ("mode", "concept", "new", "value", "note")}
    # Step 1 is preset from what is known: the cluster's concept, else a
    # listed concept matching (or spelled like) its automatic value
    match = c.get("concept") or store.find_concept(auto)
    similar = [] if match or not auto else store.similar_concepts(auto)
    preset = match or (similar[0] if similar else None)

    with st.container(border=True):
        st.markdown("**Label**")
        st.radio("1 · Does this cluster belong to a concept in the list?",
                 (EXISTING, NEW) if concepts else (NEW,), horizontal=True, key=keys["mode"],
                 index=0,                  # check the list first, if there is one
                 help=f"{len(concepts)} concept(s) in the list")
        if st.session_state[keys["mode"]] == EXISTING:
            st.selectbox("Concept", concepts, key=keys["concept"],
                         index=concepts.index(preset) if preset in concepts else None,
                         placeholder="check the list…")
            if similar and not match:
                st.caption(f"spelled like the automatic value: {', '.join(similar)}")
            concept_group(store, st.session_state[keys["concept"]])
            if st.session_state[keys["concept"]] is None:
                st.caption(f"Not in the list? Choose {NEW}.")
        else:
            st.text_input("New concept", auto or "", key=keys["new"],
                          placeholder="name of the new concept",
                          help="Added to the concept list when you save")
            typed = st.session_state[keys["new"]]
            if store.find_concept(typed):
                st.warning(f"'{store.find_concept(typed)}' is already in the list - "
                           f"choose {EXISTING}.")
        hint = f"automatic: {auto} ({c['auto_votes']})" if auto else \
            (f"automatic: {c['auto_votes']}" if c.get("auto_votes") else "no automatic value")
        st.text_input("2 · Signifier (value of this cluster)",
                      c["name"] or auto or "", key=keys["value"], help=hint,
                      placeholder="type the word this cluster shows")
        st.text_input("Note", c.get("note") or "", key=keys["note"])
        st.button("Save", type="primary", key=f"save_{tag}", on_click=save_label,
                  args=(store, run, c, annotator, keys))

    with st.popover("Remove cluster", **FULL):
        st.text_input("Reason", key=f"reason_{cid}")
        st.button("Confirm removal", key=f"rmc_{cid}", type="primary",
                  on_click=lambda: do(store, run, f"cluster #{c['number']} removed", store.remove,
                                      cid, annotator, st.session_state[f"reason_{cid}"] or None))
    with st.popover("Merge into another cluster", **FULL):
        targets = target_options(store, run, cid)
        if targets:
            st.selectbox("Target cluster", list(targets), format_func=targets.get,
                         key=f"merge_{cid}")
            st.button("Confirm merge", key=f"mg_{cid}", type="primary",
                      on_click=lambda: do(store, run, f"cluster #{c['number']} merged into "
                                          f"{targets[st.session_state[f'merge_{cid}']].split(' ')[0]}",
                                          store.merge, cid, st.session_state[f"merge_{cid}"],
                                          annotator))
    manage_concepts(store, run, annotator)
    if c["status"] == "labelled":
        st.button("Unassign (back to pending)", key=f"unl_{cid}", **FULL, on_click=do,
                  args=(store, run, f"cluster #{c['number']} back to pending", store.unassign,
                        cid, annotator))


def compare_panel(store: ClusterStore, run: str, target_ids: list[int]) -> None:
    """Words of the clusters suggested as better homes, for comparison."""
    for tid in target_ids:
        t = store.show(tid)
        occ = store.occurrences(tid, include_removed=False)
        with st.expander(f"Compare with #{t['number']} · {ClusterStore.label_text(t)} "
                         f"({len(occ)} words)"):
            cols = st.columns(6)
            for k, o in enumerate(occ):
                cols[k % 6].image(word_image(store, run, o, False),
                                  caption=f"occ {o.occ_id}", **FULL)


def cluster_view(store: ClusterStore, run: str, info: dict) -> None:
    cid = st.session_state.get("cid")
    if cid is None:
        return
    annotator = st.session_state.get("annotator", "").strip()
    c = next(x for x in store.list_clusters(run) if x["cluster_id"] == cid)
    checks = {k.occ_id: k for k in info["checks"] if k.cluster_id == cid}
    n_flag = sum(k.recommendation in ("move", "remove") for k in checks.values())

    st.header(f"Cluster #{c['number']}")
    z = f" · best z {c['best_z']:.1f}" if c["best_z"] is not None else " · created by annotator"
    st.caption(f"{c['status']} · {ClusterStore.label_text(c)} · {c['size']} words on "
               f"{c['n_pages']} page(s){z}")
    if not annotator:
        st.warning("Enter your name as annotator in the sidebar to make changes.")

    left, right = st.columns([3, 1], gap="large")
    with right:
        if annotator:
            label_panel(store, run, c, annotator)
    with left:
        if c["status"] == "removed":
            st.info("Words of a removed cluster are shown for reference only.")
        elif n_flag and annotator:
            st.button(f"Accept all {n_flag} suggestion(s) for this cluster", key=f"all_{cid}",
                      on_click=do, args=(store, run, "suggestions applied", review.apply_checks,
                                         store, run, cid, annotator), kwargs=info["thr"])
        occ = store.occurrences(cid, include_removed=False)
        # flagged words first, so the annotator sees them at once
        order = {"move": 0, "remove": 1, "single": 2, "keep": 3}
        occ.sort(key=lambda o: order[checks[o.occ_id].recommendation] if o.occ_id in checks else 4)
        readings = recognition.load_readings(store)
        targets = target_options(store, run, cid)
        editable = annotator and c["status"] != "removed"
        per_row = st.session_state.get("per_row", 3)
        for r in range(0, len(occ), per_row):
            cols = st.columns(per_row)
            for col, o in zip(cols, occ[r:r + per_row]):
                with col:
                    if editable:
                        word_tile(store, run, cid, o, checks.get(o.occ_id), readings,
                                  targets, annotator)
                    else:
                        st.image(word_image(store, run, o, st.session_state.get("context")),
                                 caption=f"occ {o.occ_id} · {verdict_text(checks.get(o.occ_id))}",
                                 **FULL)

        suggested = list(dict.fromkeys(k.best_cluster for k in checks.values()
                                       if k.recommendation == "move"))
        if suggested:
            compare_panel(store, run, suggested)

        removed = [o for o in store.occurrences(cid) if o.removed]
        if removed and editable:
            with st.expander(f"Removed words ({len(removed)})"):
                for o in removed:
                    i, b = st.columns([4, 1])
                    i.image(word_image(store, run, o, False), caption=f"occ {o.occ_id}",
                            **FULL)
                    b.button("Restore", key=f"rs_{o.occ_id}", on_click=do,
                             args=(store, run, f"occ {o.occ_id} restored",
                                   store.remove_occurrence, cid, o.occ_id, annotator),
                             kwargs={"restore": True})


# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Word clusters", layout="wide")
    folder = os.environ.get("ANNOTATION_DIR", str(HERE / "result" / "annotations"))
    store = get_store(folder)
    show_flash()
    if not store.runs():
        import_form(store)
        return
    run, info = sidebar(store)
    cluster_view(store, run, info)


main()