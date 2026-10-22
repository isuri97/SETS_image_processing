"""
Annotation backend for word-spotting clusters (annotation.py).

Takes the output of cross_spotting.run_spotting() (clusters.csv plus the
page images) and lets an annotator review every cluster and decide one of:

    concept    - label the cluster as a concept
    signifier  - label the cluster as a signifier
    remove     - reject the cluster (false positive, flourish, noise)

Names are filled in automatically: `auto` reads every word with TrOCR
(recognition.py) and gives each cluster the reading most of its words
agree on. The annotator decides concept / signifier and either keeps the
automatic name or types one by hand:

    label 9 concept             -> concept, automatic name   (name_source auto)
    label 9 concept "Padova"    -> concept, name typed by hand (manual)
    name 9 "Padova"             -> correct the name of a labelled cluster

Mixed clusters can be cleaned first: an occurrence that belongs to another
cluster is moved there (or into a new cluster), and one that is not a word
of any cluster is removed. Two clusters of the same word can be merged.
review.py flags words that seem to be in the wrong cluster.

No user interface is included: every action is a Python call (or a CLI
command), and images for display are returned as arrays / written as JPEGs,
so any frontend (Streamlit, notebook, web page) can sit on top.

Files
-----
    app.py          Streamlit app to view and annotate the clusters
    annotation.py   this file: storage, actions, sheets, export, command line
    review.py       checks whether each word fits its cluster (flags)
    recognition.py  TrOCR reading and automatic names

Storage
-------
A folder of CSV files (e.g. result/annotations), for one annotator at a
time. They can be opened in Excel to look at the state; make changes with
the commands, so they are checked and logged.

    runs.csv            imported run_spotting() output folders
    pages.csv           page name -> image path
    cluster_labels.csv  one row per cluster: status (pending / labelled /
                        removed / merged), kind (concept / signifier), name,
                        name_source (auto / manual), auto_name, auto_votes,
                        note, annotator, time - the annotation result
    occurrences.csv     one row per word: its current cluster, page, box,
                        and whether it was removed
    readings.csv        TrOCR reading of every word (recognition.py)
    history.csv         every action: who, when, what
    features.npz        word embeddings for the checks (review.py)

Every action rewrites the files through a temporary file, so a crash never
leaves a half-written file.

Python use
----------
    from core.comparing_all_similar_words.annotation import ClusterStore
    from core.comparing_all_similar_words import recognition

    store = ClusterStore("result/annotations")
    run = store.import_run("result", image_dir="data")      # once per run
    recognition.auto_name_clusters(store, run, recognition.TrOCRRecogniser())

    cid = store.cluster_id(run, 9)                          # cluster number -> id
    store.label(cid, "concept", annotator="isuri")          # automatic name
    store.label(cid, "concept", "Padova", annotator="isuri")  # typed name
    store.move_occurrence(occ_id, into=other_cid, annotator="isuri")
    store.remove(other_cid, annotator="isuri", reason="flourish")
    store.export_csv("labels.csv", run)

Command line (from the project root)
------------------------------------
    A="python -m core.comparing_all_similar_words.annotation --dir <result>/annotations"
    $A import <result> --image_dir data
    $A auto                                  # TrOCR names every cluster
    $A review                                # sheets: flags + automatic names
    $A list [--status pending]
    $A label 9 concept --by isuri            # keep the automatic name
    $A label 9 concept "Padova" --by isuri   # or type the name
    $A label 9 signifier --no_auto --by isuri  # no name at all
    $A name 9 "Padova" --by isuri            # correct a name later
    $A remove 7 --reason flourish --by isuri
    $A move 41 --to 10 --by isuri            # occ 41 -> cluster 10
    $A move 41 57 --new --by isuri           # new cluster from occs
    $A drop-occ 41 --by isuri                # not a word of any cluster
    $A merge 31 --into 6 --by isuri
    $A check [9 10]                          # misfits: move / remove?
    $A apply-checks 9 --by isuri             # accept the flags of a cluster
    $A progress
    $A export labels.csv                     # or labels.jsonld

After moving words, run `auto` again: the names follow the clusters' new
members, and only words not read before go through TrOCR.

Cluster numbers are those of clusters.csv / the sheets; occurrence numbers
("occ 41") are printed on every tile.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

KINDS = ("concept", "signifier")          # allowed labels for a cluster
STATUSES = ("pending", "labelled", "removed", "merged")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Files of the annotation folder and their columns (name, type). Empty cells
# are None. cluster_labels.csv / occurrences.csv are rewritten after every
# action; history.csv is only appended to.
TABLES = {
    "runs": ("runs.csv", [("run_id", str), ("out_dir", str), ("imported_at", str)]),
    "pages": ("pages.csv", [("run_id", str), ("name", str), ("path", str)]),
    "clusters": ("cluster_labels.csv", [
        ("cluster_id", int), ("run_id", str), ("number", int), ("best_z", float),
        ("status", str), ("kind", str), ("name", str), ("note", str),
        ("merged_into", int), ("annotator", str), ("updated_at", str),
        ("name_source", str),         # auto (TrOCR name accepted) / manual (typed)
        ("auto_name", str),           # automatic name suggestion (recognition.py)
        ("auto_votes", str)]),        # e.g. "3/4": words whose reading agrees
    "occurrences": ("occurrences.csv", [
        ("occ_id", int), ("cluster_id", int), ("origin_cluster", int), ("page", str),
        ("token", str), ("candidate", str), ("x0", int), ("y0", int), ("x1", int),
        ("y1", int), ("removed", int)]),
}
HISTORY = ("history.csv", ["event_id", "ts", "annotator", "action", "cluster_id", "payload"])


class AnnotationError(ValueError):
    """An action that is not allowed in the cluster's current state."""


@dataclass
class Occurrence:
    occ_id: int
    cluster_id: int
    page: str
    token: str
    candidate: str
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 in page pixels
    removed: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def _read_table(path: Path, cols: list[tuple[str, type]]) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{c: (t(r[c]) if r.get(c, "") != "" else None) for c, t in cols}
                for r in csv.DictReader(f)]


def _write_atomic(path: Path, header: list[str], rows: list[list]) -> None:
    """Write to a temporary file, then swap it in, so a crash never leaves a
    half-written file behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class ClusterStore:
    """All annotation state, kept as CSV files in one folder (for one
    annotator at a time):

        runs.csv            imported run_spotting() output folders
        pages.csv           page name -> image path
        cluster_labels.csv  one row per cluster: status, kind, name, note,
                            annotator, time - the annotation result
        occurrences.csv     one row per word: its cluster, page, box, and
                            whether it was removed
        history.csv         every action, with annotator and time

    The files can be opened in Excel to look at the state; make changes
    with the commands, so they are checked and logged."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if self.folder.suffix == ".db":
            raise AnnotationError(f"{folder}: annotations are now a folder of CSV files, "
                                  f"not a .db file - give a folder name")
        self.folder.mkdir(parents=True, exist_ok=True)
        self._load()
        self._page_cache: dict[str, np.ndarray] = {}
        self._pending_events: list[list] = []

    def _load(self) -> None:
        t = {k: _read_table(self.folder / f, cols) for k, (f, cols) in TABLES.items()}
        self._runs = t["runs"]
        self._pages = {(p["run_id"], p["name"]): p["path"] for p in t["pages"]}
        self._clusters = {c["cluster_id"]: c for c in t["clusters"]}
        self._occ = {o["occ_id"]: o for o in t["occurrences"]}
        hist = self.folder / HISTORY[0]
        self._history = list(csv.DictReader(open(hist, newline="", encoding="utf-8"))) \
            if hist.is_file() else []

    def close(self) -> None:
        """Nothing to close (kept so existing code keeps working)."""

    def reload(self) -> None:
        """Re-read the files, e.g. after another program (the command line,
        a second app tab) changed them. Page images stay cached."""
        self._load()

    def _save(self) -> None:
        for key in ("runs", "clusters", "occurrences"):
            fname, cols = TABLES[key]
            data = {"runs": self._runs, "clusters": self._clusters.values(),
                    "occurrences": self._occ.values()}[key]
            _write_atomic(self.folder / fname, [c for c, _ in cols],
                          [["" if r[c] is None else r[c] for c, _ in cols] for r in data])
        fname, cols = TABLES["pages"]
        _write_atomic(self.folder / fname, [c for c, _ in cols],
                      [[r, n, p] for (r, n), p in self._pages.items()])
        if self._pending_events:
            hist = self.folder / HISTORY[0]
            new = not hist.is_file()
            with open(hist, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(HISTORY[1])
                w.writerows(self._pending_events)
            self._history += [dict(zip(HISTORY[1], map(str, e))) for e in self._pending_events]
            self._pending_events = []

    @contextmanager
    def _tx(self) -> Iterator[None]:
        """One action = one save. If anything fails, the in-memory state is
        rolled back and nothing is written."""
        snapshot = (copy.deepcopy(self._runs), dict(self._pages),
                    copy.deepcopy(self._clusters), copy.deepcopy(self._occ))
        try:
            yield
            self._save()
        except BaseException:
            self._runs, self._pages, self._clusters, self._occ = snapshot
            self._pending_events = []
            raise

    def _log(self, action: str, cluster_id: int | None, annotator: str | None,
             **payload) -> None:
        eid = len(self._history) + len(self._pending_events) + 1
        self._pending_events.append([eid, _now(), annotator or "", action,
                                     "" if cluster_id is None else cluster_id,
                                     json.dumps(payload, ensure_ascii=False)])

    def _touch(self, c: dict, annotator: str | None, **fields) -> None:
        c.update(fields, annotator=annotator, updated_at=_now())

    # ------------------------------------------------------------------ #
    # Import
    # ------------------------------------------------------------------ #
    def import_run(self, out_dir: str | Path, image_dir: str | Path | None = None,
                   run_id: str | None = None) -> str:
        """Load clusters.csv of a run_spotting() output folder.

        `image_dir` is where the page images are (searched by page name and
        IMAGE_EXTS); defaults to `out_dir`. `run_id` defaults to the folder
        name. A run is imported once; importing the same run_id again is
        refused so annotations are never overwritten."""
        out_dir = Path(out_dir)
        csv_path = out_dir / "clusters.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"{csv_path} not found - run run_spotting() first")
        run_id = run_id or out_dir.resolve().name
        if run_id in self.runs():
            raise AnnotationError(f"run '{run_id}' is already imported; pass another run_id")

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        page_paths = self._find_pages({r["page"] for r in rows}, Path(image_dir or out_dir))

        with self._tx():
            self._runs.append(dict(run_id=run_id, out_dir=str(out_dir.resolve()),
                                   imported_at=_now()))
            self._pages.update({(run_id, n): str(p) for n, p in page_paths.items()})
            next_cid = max(self._clusters, default=0) + 1
            next_occ = max(self._occ, default=0) + 1
            ids: dict[int, int] = {}
            for r in rows:
                num = int(r["cluster"])
                if num not in ids:
                    ids[num] = next_cid
                    self._clusters[next_cid] = dict(
                        cluster_id=next_cid, run_id=run_id, number=num,
                        best_z=float(r["best_z"]), status="pending", kind=None, name=None,
                        note=None, merged_into=None, annotator=None, updated_at=None,
                        name_source=None, auto_name=None, auto_votes=None)
                    next_cid += 1
                self._occ[next_occ] = dict(
                    occ_id=next_occ, cluster_id=ids[num], origin_cluster=ids[num],
                    page=r["page"], token=r["token"], candidate=r["candidate"],
                    x0=int(r["x0"]), y0=int(r["y0"]), x1=int(r["x1"]), y1=int(r["y1"]),
                    removed=0)
                next_occ += 1
            self._log("import", None, None, run_id=run_id, clusters=len(ids),
                      occurrences=len(rows))
        return run_id

    @staticmethod
    def _find_pages(names: set[str], image_dir: Path) -> dict[str, Path]:
        found = {}
        for name in names:
            hits = [image_dir / f"{name}{ext}" for ext in IMAGE_EXTS
                    if (image_dir / f"{name}{ext}").is_file()]
            if not hits:
                raise FileNotFoundError(f"no image for page '{name}' in {image_dir}")
            found[name] = hits[0]
        return found

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def runs(self) -> list[str]:
        return [r["run_id"] for r in self._runs]

    def cluster_id(self, run_id: str, number: int) -> int:
        """Internal id of cluster `number` (as in clusters.csv/.jpg) of a run."""
        for c in self._clusters.values():
            if c["run_id"] == run_id and c["number"] == number:
                return c["cluster_id"]
        raise AnnotationError(f"run '{run_id}' has no cluster {number}")

    def _cluster_row(self, cluster_id: int) -> dict:
        if cluster_id not in self._clusters:
            raise AnnotationError(f"no cluster {cluster_id}")
        return self._clusters[cluster_id]

    @staticmethod
    def _as_occurrence(o: dict) -> Occurrence:
        return Occurrence(o["occ_id"], o["cluster_id"], o["page"], o["token"], o["candidate"],
                          (o["x0"], o["y0"], o["x1"], o["y1"]), bool(o["removed"]))

    def occurrences(self, cluster_id: int, include_removed: bool = True) -> list[Occurrence]:
        occ = [o for o in self._occ.values() if o["cluster_id"] == cluster_id
               and (include_removed or not o["removed"])]
        occ.sort(key=lambda o: (o["page"], o["y0"], o["x0"]))
        return [self._as_occurrence(o) for o in occ]

    def active_members(self, run_id: str, statuses: Sequence[str] = ("pending", "labelled")
                       ) -> list[tuple[Occurrence, dict]]:
        """(word, its cluster) for every non-removed word of the run's
        clusters with one of `statuses`, in cluster-number order."""
        out = [(self._as_occurrence(o), self._clusters[o["cluster_id"]])
               for o in self._occ.values()
               if not o["removed"] and self._clusters[o["cluster_id"]]["run_id"] == run_id
               and self._clusters[o["cluster_id"]]["status"] in statuses]
        return sorted(out, key=lambda m: (m[1]["number"], m[0].occ_id))

    def run_occurrences(self, run_id: str) -> list[Occurrence]:
        """Every word imported for the run, removed or not."""
        return [self._as_occurrence(o) for o in self._occ.values()
                if self._clusters[o["origin_cluster"]]["run_id"] == run_id]

    def list_clusters(self, run_id: str, status: str | None = None,
                      kind: str | None = None, unnamed: bool = False) -> list[dict]:
        """Clusters of a run in their original order, optionally filtered
        (unnamed=True: labelled clusters that still have no name)."""
        size: Counter = Counter()
        pages: dict[int, set] = {}
        for o in self._occ.values():
            if not o["removed"]:
                size[o["cluster_id"]] += 1
                pages.setdefault(o["cluster_id"], set()).add(o["page"])
        out = []
        for c in self._clusters.values():
            if c["run_id"] != run_id or (status and c["status"] != status) \
                    or (kind and c["kind"] != kind) \
                    or (unnamed and not (c["status"] == "labelled" and c["name"] is None)):
                continue
            out.append(dict(c, size=size[c["cluster_id"]],
                            n_pages=len(pages.get(c["cluster_id"], ()))))
        return sorted(out, key=lambda c: c["number"])

    def history(self, cluster_id: int | None = None) -> list[dict]:
        """Logged actions (of one cluster, or all), oldest first."""
        return [dict(h) for h in self._history
                if cluster_id is None or h["cluster_id"] == str(cluster_id)]

    def show(self, cluster_id: int) -> dict:
        """Everything a frontend needs to display one cluster."""
        c = dict(self._cluster_row(cluster_id))
        c["occurrences"] = [o.__dict__ for o in self.occurrences(cluster_id)]
        c["history"] = [{k: h[k] for k in ("ts", "annotator", "action", "payload")}
                        for h in self.history(cluster_id)]
        return c

    def next_pending(self, run_id: str, after: int | None = None) -> int | None:
        """Id of the next pending cluster (in cluster order), or None when done."""
        start = self._cluster_row(after)["number"] if after is not None else None
        for c in self.list_clusters(run_id, status="pending"):
            if start is None or c["number"] > start:
                return c["cluster_id"]
        return None

    def progress(self, run_id: str) -> dict[str, int]:
        counts = {s: 0 for s in STATUSES} | {k: 0 for k in KINDS}
        for c in self.list_clusters(run_id):
            counts[c["status"]] += 1
            if c["kind"]:
                counts[c["kind"]] += 1
        counts["total"] = sum(counts[s] for s in STATUSES)
        counts["unnamed"] = len(self.list_clusters(run_id, unnamed=True))
        open_ = [c for c in self.list_clusters(run_id) if c["status"] in ("pending", "labelled")]
        counts["auto_named"] = sum(1 for c in open_ if c.get("auto_name"))
        counts["name_auto"] = sum(1 for c in open_ if c.get("name_source") == "auto")
        counts["name_manual"] = sum(1 for c in open_ if c.get("name_source") == "manual")
        return counts

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _require_open(self, cluster_id: int) -> dict:
        row = self._cluster_row(cluster_id)
        if row["status"] in ("removed", "merged"):
            raise AnnotationError(f"cluster {cluster_id} is {row['status']}; restore it first")
        return row

    def label(self, cluster_id: int, kind: str, name: str | None = None,
              annotator: str | None = None, note: str | None = None,
              use_auto: bool = True) -> None:
        """Define the cluster as a concept or a signifier (always the
        annotator's decision).

        The name:
          - given (e.g. "Padova")         -> used as typed (name_source manual)
          - omitted, automatic name exists -> the automatic name is used
                                              (name_source auto), unless
                                              use_auto=False
          - omitted, no automatic name     -> the cluster stays unnamed; a
                                              name can be added with name()
        Re-labelling is allowed and logged."""
        if kind not in KINDS:
            raise AnnotationError(f"kind must be one of {KINDS}, not '{kind}'")
        if not annotator:
            raise AnnotationError("annotator is required")
        name = name.strip() if name and name.strip() else None
        c = self._require_open(cluster_id)
        if not self.occurrences(cluster_id, include_removed=False):
            raise AnnotationError(f"cluster {cluster_id} has no remaining occurrences")
        source = "manual" if name else None
        if name is None and use_auto and c.get("auto_name"):
            name, source = c["auto_name"], "auto"
        with self._tx():
            previous = {"kind": c["kind"], "name": c["name"]}
            self._touch(c, annotator, status="labelled", kind=kind, name=name, note=note,
                        name_source=source)
            self._log("label", cluster_id, annotator, kind=kind, name=name,
                      name_source=source, note=note, previous=previous)

    def name(self, cluster_id: int, name: str | None, annotator: str) -> None:
        """Add, change or clear (None) the name of a labelled cluster by hand,
        without touching its kind - e.g. correcting an automatic name."""
        c = self._require_open(cluster_id)
        if c["status"] != "labelled":
            raise AnnotationError(f"cluster {cluster_id} is {c['status']}; label it first")
        name = name.strip() if name and name.strip() else None
        with self._tx():
            previous = c["name"]
            self._touch(c, annotator, name=name, name_source="manual" if name else None)
            self._log("name", cluster_id, annotator, name=name, previous=previous)

    def set_auto_names(self, names: dict[int, tuple[str | None, str | None]],
                       source: str = "trocr") -> int:
        """Store automatic name suggestions: cluster_id -> (name, votes),
        e.g. ('Padova', '3/4'); (None, None) clears one. Only the
        suggestion columns change - never a name the annotator has set.
        Returns the number of clusters whose suggestion changed."""
        changed = {cid: v for cid, v in names.items()
                   if (self._cluster_row(cid).get("auto_name"),
                       self._cluster_row(cid).get("auto_votes")) != tuple(v)}
        if not changed:
            return 0
        with self._tx():
            for cid, (auto, votes) in changed.items():
                self._clusters[cid].update(auto_name=auto, auto_votes=votes)
                self._log("auto_name", cid, source, auto_name=auto, votes=votes)
        return len(changed)

    def remove(self, cluster_id: int, annotator: str, reason: str | None = None) -> None:
        """Reject the whole cluster. Reversible with restore()."""
        c = self._require_open(cluster_id)
        with self._tx():
            self._touch(c, annotator, status="removed", note=reason)
            self._log("remove", cluster_id, annotator, reason=reason)

    def restore(self, cluster_id: int, annotator: str) -> None:
        """Undo remove(): the cluster returns to its label, or to pending."""
        c = self._cluster_row(cluster_id)
        if c["status"] != "removed":
            raise AnnotationError(f"cluster {cluster_id} is {c['status']}, not removed")
        status = "labelled" if c["kind"] else "pending"
        with self._tx():
            self._touch(c, annotator, status=status)
            self._log("restore", cluster_id, annotator, status=status)

    def unlabel(self, cluster_id: int, annotator: str) -> None:
        """Send a labelled cluster back to pending."""
        c = self._require_open(cluster_id)
        with self._tx():
            self._touch(c, annotator, status="pending", kind=None, name=None,
                        name_source=None)
            self._log("unlabel", cluster_id, annotator)

    def remove_occurrence(self, cluster_id: int, occ_id: int, annotator: str,
                          restore: bool = False) -> None:
        """Take one wrong member out of a cluster (or put it back with
        restore=True) without rejecting the cluster."""
        self._require_open(cluster_id)
        o = self._occ.get(occ_id)
        if o is None or o["cluster_id"] != cluster_id:
            raise AnnotationError(f"occurrence {occ_id} is not in cluster {cluster_id}")
        with self._tx():
            o["removed"] = 0 if restore else 1
            self._log("restore_occurrence" if restore else "remove_occurrence",
                      cluster_id, annotator, occ_id=occ_id)

    def merge(self, cluster_id: int, into: int, annotator: str) -> None:
        """Move all occurrences of `cluster_id` into cluster `into` (the same
        word split over two clusters). The source cluster becomes 'merged';
        the target keeps its own status and label."""
        if cluster_id == into:
            raise AnnotationError("cannot merge a cluster into itself")
        src, dst = self._require_open(cluster_id), self._require_open(into)
        if src["run_id"] != dst["run_id"]:
            raise AnnotationError("clusters belong to different runs")
        with self._tx():
            for o in self._occ.values():
                if o["cluster_id"] == cluster_id:
                    o["cluster_id"] = into
            self._touch(src, annotator, status="merged", merged_into=into)
            self._log("merge", cluster_id, annotator, into=into)
            self._log("merge_target", into, annotator, source=cluster_id)

    def cluster_of(self, occ_id: int) -> int:
        """Id of the cluster an occurrence currently belongs to."""
        if occ_id not in self._occ:
            raise AnnotationError(f"no occurrence {occ_id}")
        return self._occ[occ_id]["cluster_id"]

    def move_occurrence(self, occ_id: int, into: int, annotator: str) -> None:
        """Move one occurrence to another cluster (e.g. a word in cluster 9
        that belongs to cluster 10). The target keeps its status and label;
        a previously removed occurrence becomes active again."""
        src = self.cluster_of(occ_id)
        if src == into:
            raise AnnotationError(f"occurrence {occ_id} is already in cluster {into}")
        s, d = self._require_open(src), self._require_open(into)
        if s["run_id"] != d["run_id"]:
            raise AnnotationError("clusters belong to different runs")
        with self._tx():
            self._occ[occ_id].update(cluster_id=into, removed=0)
            self._log("move_out", src, annotator, occ_id=occ_id, to=into)
            self._log("move_in", into, annotator, occ_id=occ_id, source=src)

    def new_cluster(self, occ_ids: list[int], annotator: str) -> int:
        """Create a new pending cluster from one or more occurrences (taken
        out of their current clusters). Returns the new cluster's id; its
        number is the next free number of the run."""
        if not occ_ids:
            raise AnnotationError("give at least one occurrence")
        srcs = [self._require_open(self.cluster_of(o)) for o in occ_ids]
        run_id = srcs[0]["run_id"]
        if any(s["run_id"] != run_id for s in srcs):
            raise AnnotationError("occurrences belong to different runs")
        with self._tx():
            number = max(c["number"] for c in self._clusters.values()
                         if c["run_id"] == run_id) + 1
            new = max(self._clusters) + 1
            self._clusters[new] = dict(
                cluster_id=new, run_id=run_id, number=number, best_z=None,
                status="pending", kind=None, name=None, note=None, merged_into=None,
                annotator=annotator, updated_at=_now(), name_source=None,
                auto_name=None, auto_votes=None)
            for o, s in zip(occ_ids, srcs):
                self._occ[o].update(cluster_id=new, removed=0)
                self._log("move_out", s["cluster_id"], annotator, occ_id=o, to=new)
            self._log("create", new, annotator, occ_ids=list(occ_ids))
        return new

    # ------------------------------------------------------------------ #
    # Carry labels over to a re-run
    # ------------------------------------------------------------------ #
    def carry_over(self, from_run: str, to_run: str, annotator: str,
                   min_iou: float = 0.5, min_share: float = 0.5) -> int:
        """Apply decisions of `from_run` to still-pending clusters of
        `to_run`. A new cluster inherits an old decision (label or removal)
        if at least `min_share` of its occurrences overlap (box IoU >=
        `min_iou`, same page) occurrences of one decided old cluster.
        Returns the number of clusters updated; each update is logged."""
        old = {}
        for c in self.list_clusters(from_run):
            if c["status"] in ("labelled", "removed"):
                old[c["cluster_id"]] = (c, self.occurrences(c["cluster_id"], False))
        n = 0
        for c in self.list_clusters(to_run, status="pending"):
            occ = self.occurrences(c["cluster_id"], False)
            best, best_share = None, 0.0
            for oc, oocc in old.values():
                hits = sum(any(o.page == p.page and _iou(o.box, p.box) >= min_iou
                               for p in oocc) for o in occ)
                share = hits / len(occ) if occ else 0.0
                if share > best_share:
                    best, best_share = oc, share
            if best is None or best_share < min_share:
                continue
            note = f"carried over from {from_run} cluster {best['number']} ({best_share:.0%})"
            if best["status"] == "removed":
                self.remove(c["cluster_id"], annotator, reason=note)
            else:
                self.label(c["cluster_id"], best["kind"], best["name"], annotator, note=note)
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Images for display
    # ------------------------------------------------------------------ #
    def _page(self, run_id: str, name: str) -> np.ndarray:
        key = f"{run_id}/{name}"
        if key not in self._page_cache:
            path = self._pages.get((run_id, name))
            if path is None:
                raise AnnotationError(f"page '{name}' is not part of run '{run_id}'")
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"cannot read page image {path}")
            self._page_cache[key] = img
        return self._page_cache[key]

    def crop(self, occ: Occurrence, run_id: str, context: bool = True,
             pad_lines: float = 0.6, pad_words: float = 3.0,
             margin: float = 0.0) -> np.ndarray:
        """The occurrence on its page. With context=True the crop also shows
        the surrounding text (`pad_words` box heights left and right,
        `pad_lines` box heights above and below) and the word is boxed.
        With context=False only the word, plus `margin` x its height of page
        around it (recognisers expect a little margin)."""
        page = self._page(run_id, occ.page)
        x0, y0, x1, y1 = occ.box
        if not context:
            m = int(margin * (y1 - y0))
            return page[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m].copy()
        h = y1 - y0
        X0, Y0 = max(0, int(x0 - pad_words * h)), max(0, int(y0 - pad_lines * h))
        X1 = min(page.shape[1], int(x1 + pad_words * h))
        Y1 = min(page.shape[0], int(y1 + pad_lines * h))
        out = page[Y0:Y1, X0:X1].copy()
        cv2.rectangle(out, (x0 - X0, y0 - Y0), (x1 - X0, y1 - Y0), (0, 0, 220), 2)
        return out

    def render_cluster(self, cluster_id: int, context: bool = True,
                       width: int = 700, row_h: int = 110) -> np.ndarray:
        """A display sheet: header with status and label, then one row per
        occurrence (context crop, page, occurrence id). Removed occurrences
        are greyed and struck through."""
        c = self._cluster_row(cluster_id)
        head = np.full((34, width, 3), 235, np.uint8)
        cv2.putText(head, self._title(c), (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        rows = [head]
        for o in self.occurrences(cluster_id):
            img = self.crop(o, c["run_id"], context=context)
            s = min((row_h - 24) / img.shape[0], width / img.shape[1])
            img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
            row = np.full((row_h, width, 3), 255, np.uint8)
            row[22:22 + img.shape[0], :img.shape[1]] = img
            cv2.putText(row, f"occ {o.occ_id}  {o.page}:{o.token}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if o.removed:
                row = cv2.addWeighted(row, 0.35, np.full_like(row, 255), 0.65, 0)
                cv2.line(row, (0, row_h // 2), (width, row_h // 2), (0, 0, 200), 2)
                cv2.putText(row, "removed", (width - 90, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 200), 1)
            rows.append(row)
        return cv2.vconcat(rows)

    @staticmethod
    def label_text(c) -> str:
        """Short description of a cluster's label, e.g. 'concept: Padova
        (auto)', 'unlabelled - auto: Padova 3/4'."""
        if c["kind"]:
            if not c["name"]:
                return f"{c['kind']} (unnamed)"
            src = f" ({c['name_source']})" if c.get("name_source") else ""
            return f"{c['kind']}: {c['name']}{src}"
        if c.get("auto_name"):
            return f"unlabelled - auto: {c['auto_name']} {c.get('auto_votes') or ''}".rstrip()
        if c.get("auto_votes"):                      # read, but readings disagree
            return f"unlabelled - auto: {c['auto_votes']}"
        return "unlabelled"

    @classmethod
    def _title(cls, c) -> str:
        z = f"  best z {c['best_z']:.1f}" if c["best_z"] is not None else "  (created by annotator)"
        return f"cluster {c['number']} [{c['status']}] {cls.label_text(c)}{z}"

    def render_overview(self, cluster_ids: list[int], per_row: int = 6,
                        tile_w: int = 220, tile_h: int = 80,
                        show_removed: bool = False,
                        marks: dict[int, tuple[str, tuple[int, int, int]]] | None = None,
                        notes: dict[int, str] | None = None,
                        extra: dict[int, str] | None = None) -> np.ndarray:
        """Many clusters on one sheet, so they can be compared: one block per
        cluster (header with number, status and label), then its occurrences
        as tiles labelled with the occurrence id and page. The occurrence id
        is what move / drop commands take.

        Optional overlays (used by review.py):
          marks : occ_id -> (text, BGR colour); the tile gets a coloured frame
                  and the text, e.g. ("-> #10", orange)
          notes : occ_id -> short text under the word
          extra : cluster_id -> text appended to the cluster header"""
        marks, notes, extra = marks or {}, notes or {}, extra or {}
        blocks = []
        width = per_row * (tile_w + 8)
        for cid in cluster_ids:
            c = self._cluster_row(cid)
            occ = [o for o in self.occurrences(cid) if show_removed or not o.removed]
            head = np.full((30, width, 3), 225, np.uint8)
            cv2.putText(head, self._title(c) + f"  ({len(occ)} occ)" + extra.get(cid, ""),
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            foot = 18 if notes else 0
            th, tw = tile_h + 22 + foot, tile_w + 8
            tiles = []
            for o in occ:
                img = self.crop(o, c["run_id"], context=False)
                s = min(tile_h / img.shape[0], tile_w / img.shape[1])
                img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
                t = np.full((th, tw, 3), 255, np.uint8)
                y, x = 20 + (tile_h - img.shape[0]) // 2, 4 + (tile_w - img.shape[1]) // 2
                t[y:y + img.shape[0], x:x + img.shape[1]] = img
                cv2.putText(t, f"occ {o.occ_id}  {o.page}", (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 0, 0), 1)
                if o.occ_id in notes:
                    cv2.putText(t, notes[o.occ_id][:30], (4, th - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)
                if o.occ_id in marks:
                    text, col = marks[o.occ_id]
                    cv2.rectangle(t, (1, 1), (tw - 2, th - 2), col, 3)
                    (w_txt, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.putText(t, text, (tw - w_txt - 6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, col, 1)
                if o.removed:
                    t = cv2.addWeighted(t, 0.35, np.full_like(t, 255), 0.65, 0)
                tiles.append(t)
            if not tiles:
                tiles = [np.full((th, tw, 3), 255, np.uint8)]
            while len(tiles) % per_row:
                tiles.append(np.full_like(tiles[0], 255))
            rows = [cv2.hconcat(tiles[r:r + per_row]) for r in range(0, len(tiles), per_row)]
            blocks.append(cv2.vconcat([head] + rows + [np.full((10, width, 3), 255, np.uint8)]))
        return cv2.vconcat(blocks) if blocks else np.full((50, width, 3), 255, np.uint8)

    def write_overviews(self, run_id: str, out_dir: str | Path, per_sheet: int = 15,
                        status: str | None = None, cluster_ids: list[int] | None = None,
                        prefix: str = "overview", **overlays) -> list[Path]:
        """Overview sheets of all clusters of a run (optionally one status,
        or the given cluster ids), `per_sheet` clusters per image:
        overview_01.jpg, overview_02.jpg, ... `overlays` (marks / notes /
        extra) are passed to render_overview()."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        cids = cluster_ids if cluster_ids is not None else \
            [c["cluster_id"] for c in self.list_clusters(run_id, status=status)
             if c["status"] != "merged"]
        paths = []
        for k in range(0, len(cids), per_sheet):
            p = out_dir / f"{prefix}_{k // per_sheet + 1:02d}.jpg"
            cv2.imwrite(str(p), self.render_overview(cids[k:k + per_sheet], **overlays),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(p)
        return paths

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _decided(self, run_id: str, include_removed: bool):
        statuses = ("labelled", "removed") if include_removed else ("labelled",)
        for c in self.list_clusters(run_id):
            if c["status"] in statuses:
                for o in self.occurrences(c["cluster_id"], include_removed=False):
                    yield c, o

    def export_csv(self, path: str | Path, run_id: str, include_removed: bool = False) -> int:
        """One row per kept occurrence of every decided cluster."""
        n = 0
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "cluster", "status", "kind", "name", "name_source", "auto_name",
                        "page", "token", "candidate", "x0", "y0", "x1", "y1", "annotator",
                        "updated_at", "note"])
            for c, o in self._decided(run_id, include_removed):
                w.writerow([run_id, c["number"], c["status"], c["kind"] or "", c["name"] or "",
                            c.get("name_source") or "", c.get("auto_name") or "",
                            o.page, o.token, o.candidate, *o.box, c["annotator"],
                            c["updated_at"], c["note"] or ""])
                n += 1
        return n

    def export_web_annotations(self, path: str | Path, run_id: str,
                               base_uri: str = "") -> int:
        """W3C Web Annotation (JSON-LD), one annotation per kept occurrence of
        every labelled cluster. The target is the page image (`base_uri` +
        file name) with an xywh fragment selector; the bodies carry the name
        (tagging) and the kind (classifying)."""
        pages = {n: Path(p).name for (r, n), p in self._pages.items() if r == run_id}
        items = []
        for c, o in self._decided(run_id, include_removed=False):
            x0, y0, x1, y1 = o.box
            body = [{"type": "TextualBody", "purpose": "classifying", "value": c["kind"]},
                    {"type": "TextualBody", "purpose": "identifying",
                     "value": f"{run_id}/cluster/{c['number']}"}]
            if c["name"]:                       # unnamed clusters carry no tagging body
                body.insert(0, {"type": "TextualBody", "purpose": "tagging", "value": c["name"]})
            items.append({
                "id": f"urn:cluster-annotation:{run_id}:{o.occ_id}",
                "type": "Annotation",
                "motivation": "tagging" if c["name"] else "classifying",
                "creator": c["annotator"],
                "modified": c["updated_at"],
                "body": body,
                "target": {
                    "source": f"{base_uri}{pages[o.page]}",
                    "selector": {"type": "FragmentSelector",
                                 "conformsTo": "http://www.w3.org/TR/media-frags/",
                                 "value": f"xywh={x0},{y0},{x1 - x0},{y1 - y0}"},
                },
            })
        doc = {"@context": "http://www.w3.org/ns/anno.jsonld",
               "type": "AnnotationCollection", "label": f"word clusters {run_id}",
               "total": len(items), "items": items}
        Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        return len(items)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _default_run(store: ClusterStore, run: str | None) -> str:
    runs = store.runs()
    if run:
        if run not in runs:
            raise SystemExit(f"unknown run '{run}'; imported: {runs}")
        return run
    if not runs:
        raise SystemExit("no run imported yet - use the 'import' command")
    return runs[-1]                                       # latest import


def _sibling(name: str):
    """Import review.py / recognition.py from the same folder - inside a
    package (core.comparing_all_similar_words) or as plain scripts."""
    import importlib
    return importlib.import_module(f"{__package__}.{name}" if __package__ else name)


def _auto_command(store: ClusterStore, run: str, a) -> None:
    """Automatic naming with TrOCR (recognition.py)."""
    rec = _sibling("recognition")
    recogniser = rec.TrOCRRecogniser(a.model, a.device)
    lexicon = rec.load_lexicon(a.lexicon) if a.lexicon else None
    result = rec.auto_name_clusters(
        store, run, recogniser, lexicon=lexicon, recompute=a.recompute,
        progress=lambda k, t: print(f"  {k}/{t} words read", end="\r", flush=True))
    print(f"\n{result['read']} words read, {result['named']} clusters named automatically, "
          f"{result['unagreed']} without agreement (name them by hand), "
          f"{result['changed']} suggestions changed")
    for c in store.list_clusters(run, status="pending"):
        if c.get("auto_name"):
            print(f"  #{c['number']:<4} {c['auto_name']}  ({c['auto_votes']})")


def _review_command(store: ClusterStore, run: str, cid: int | None, a) -> None:
    """Commands backed by review.py (imported only when used)."""
    cr = _sibling("review")
    thr = dict(min_z=a.min_z, margin=a.margin) if hasattr(a, "min_z") else {}
    if a.cmd == "check":
        cids = [store.cluster_id(run, n) for n in a.clusters] or None
        checks = cr.check(store, run, cids, **thr)
        shown = [c for c in checks if a.all or c.recommendation in ("move", "remove")]
        for c in shown:
            print(c.describe())
        n = Counter(c.recommendation for c in checks)
        print(f"{len(checks)} words: {n['keep']} keep, {n['move']} move, "
              f"{n['remove']} remove, {n['single']} single")
    elif a.cmd == "review":
        cids = [store.cluster_id(run, n) for n in a.clusters] or None
        for p in cr.write_review_sheets(store, run, a.out, a.per_sheet, cluster_ids=cids, **thr):
            print(f"sheet written to {p}")
    elif a.cmd == "apply-checks":
        done = cr.apply_checks(store, run, cid, a.by, remove=not a.no_remove, **thr)
        print("\n".join(done) or "nothing to change")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Annotate word-spotting clusters.")
    ap.add_argument("--dir", default="annotations",
                    help="annotation folder (CSV files; created at import)")
    ap.add_argument("--run", help="run id (default: latest imported run)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import", help="import a run_spotting() output folder")
    p.add_argument("out_dir"); p.add_argument("--image_dir"); p.add_argument("--run_id")
    p = sub.add_parser("list", help="list clusters")
    p.add_argument("--status", choices=STATUSES); p.add_argument("--kind", choices=KINDS)
    p.add_argument("--unnamed", action="store_true", help="labelled clusters without a name")
    p = sub.add_parser("show", help="write a cluster sheet image")
    p.add_argument("cluster", type=int); p.add_argument("--out")
    p.add_argument("--no_context", action="store_true")
    p = sub.add_parser("next", help="id of the next pending cluster")
    p = sub.add_parser("auto", help="name every cluster automatically with TrOCR")
    p.add_argument("--model", default="microsoft/trocr-base-handwritten",
                   help="TrOCR checkpoint (Hugging Face name or local folder)")
    p.add_argument("--device", help="cpu / cuda (default: cuda if available)")
    p.add_argument("--lexicon", help="optional file of known names, one per line")
    p.add_argument("--recompute", action="store_true", help="read every word again")
    p = sub.add_parser("label", help="define a cluster as concept or signifier")
    p.add_argument("cluster", type=int); p.add_argument("kind", choices=KINDS)
    p.add_argument("name", nargs="?",
                   help="type a name to set it by hand; omit to use the automatic name")
    p.add_argument("--no_auto", action="store_true",
                   help="do not use the automatic name (leave the cluster unnamed)")
    p.add_argument("--note"); p.add_argument("--by", required=True)
    p = sub.add_parser("name", help="add or change the name of a labelled cluster by hand")
    p.add_argument("cluster", type=int); p.add_argument("name", nargs="?",
                                                        help="omit to clear the name")
    p.add_argument("--by", required=True)
    p = sub.add_parser("remove", help="reject a cluster")
    p.add_argument("cluster", type=int); p.add_argument("--reason")
    p.add_argument("--by", required=True)
    p = sub.add_parser("restore", help="undo remove")
    p.add_argument("cluster", type=int); p.add_argument("--by", required=True)
    p = sub.add_parser("overview", help="write sheets showing many clusters together")
    p.add_argument("clusters", type=int, nargs="*",
                   help="cluster numbers (default: all, split into sheets)")
    p.add_argument("--status", choices=STATUSES); p.add_argument("--per_sheet", type=int, default=15)
    p.add_argument("--out", default="overview")
    p = sub.add_parser("drop-occ", help="remove an occurrence completely (not a word of any cluster)")
    p.add_argument("occ", type=int); p.add_argument("--undo", action="store_true")
    p.add_argument("--by", required=True)
    p = sub.add_parser("move", help="move occurrences to another cluster or a new one")
    p.add_argument("occ", type=int, nargs="+")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--to", type=int, help="number of the target cluster")
    g.add_argument("--new", action="store_true", help="put them in a new cluster")
    p.add_argument("--by", required=True)
    p = sub.add_parser("merge", help="merge a cluster into another")
    p.add_argument("cluster", type=int); p.add_argument("--into", type=int, required=True)
    p.add_argument("--by", required=True)
    p = sub.add_parser("carry-over", help="apply decisions of an earlier run")
    p.add_argument("from_run"); p.add_argument("--by", required=True)
    # review assistance (review.py)
    p = sub.add_parser("check", help="does every word fit its cluster? suggest moves / removals")
    p.add_argument("clusters", type=int, nargs="*", help="cluster numbers (default: all open)")
    p.add_argument("--all", action="store_true", help="also list words that fit (keep)")
    p.add_argument("--min_z", type=float, default=1.5); p.add_argument("--margin", type=float, default=0.5)
    p = sub.add_parser("review", help="overview sheets with the check results on the tiles")
    p.add_argument("clusters", type=int, nargs="*", help="cluster numbers (default: all pending)")
    p.add_argument("--out", default="review"); p.add_argument("--per_sheet", type=int, default=10)
    p.add_argument("--min_z", type=float, default=1.5); p.add_argument("--margin", type=float, default=0.5)
    p = sub.add_parser("apply-checks", help="carry out the recommended moves/removals of a cluster")
    p.add_argument("cluster", type=int); p.add_argument("--no_remove", action="store_true")
    p.add_argument("--min_z", type=float, default=1.5); p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--by", required=True)
    sub.add_parser("progress", help="counts per status and kind")
    p = sub.add_parser("export", help="export decisions (.csv or .jsonld)")
    p.add_argument("path"); p.add_argument("--base_uri", default="")
    p.add_argument("--include_removed", action="store_true")
    a = ap.parse_args(argv)

    store = ClusterStore(a.dir)
    try:
        if a.cmd == "import":
            print(f"imported run '{store.import_run(a.out_dir, a.image_dir, a.run_id)}'")
            return
        run = _default_run(store, a.run)
        # the CLI uses cluster numbers as shown in clusters.csv / clusters.jpg
        cid = store.cluster_id(run, a.cluster) if hasattr(a, "cluster") else None
        if a.cmd == "list":
            for c in store.list_clusters(run, a.status, a.kind, a.unnamed):
                z = f"z {c['best_z']:.1f}" if c["best_z"] is not None else "new  "
                print(f"#{c['number']:<4} {c['status']:<9} {c['size']:>3} occ / "
                      f"{c['n_pages']} pages  {z}  {store.label_text(c)}")
        elif a.cmd == "overview":
            if a.clusters:
                out = Path(f"{a.out}.jpg")
                cv2.imwrite(str(out), store.render_overview(
                    [store.cluster_id(run, n) for n in a.clusters]))
                print(f"sheet written to {out}")
            else:
                for p in store.write_overviews(run, a.out, a.per_sheet, a.status):
                    print(f"sheet written to {p}")
        elif a.cmd == "show":
            out = a.out or f"cluster_{a.cluster}.jpg"
            cv2.imwrite(out, store.render_cluster(cid, context=not a.no_context))
            for o in store.occurrences(cid):
                print(f"occ {o.occ_id:>5}  {o.page}:{o.token}  box {o.box}"
                      f"{'  (removed)' if o.removed else ''}")
            print(f"sheet written to {out}")
        elif a.cmd == "next":
            nxt = store.next_pending(run)
            print(store.show(nxt)["number"] if nxt else "no pending clusters")
        elif a.cmd == "auto":
            _auto_command(store, run, a)
        elif a.cmd == "label":
            store.label(cid, a.kind, a.name, a.by, a.note, use_auto=not a.no_auto)
            print(f"cluster {a.cluster}: {store.label_text(store.show(cid))}")
        elif a.cmd == "name":
            store.name(cid, a.name, a.by)
        elif a.cmd == "remove":
            store.remove(cid, a.by, a.reason)
        elif a.cmd == "restore":
            store.restore(cid, a.by)
        elif a.cmd == "drop-occ":
            store.remove_occurrence(store.cluster_of(a.occ), a.occ, a.by, restore=a.undo)
        elif a.cmd == "move":
            if a.new:
                new = store.new_cluster(a.occ, a.by)
                print(f"new cluster {store.show(new)['number']}")
            else:
                target = store.cluster_id(run, a.to)
                for o in a.occ:
                    store.move_occurrence(o, target, a.by)
        elif a.cmd == "merge":
            store.merge(cid, store.cluster_id(run, a.into), a.by)
        elif a.cmd == "carry-over":
            print(f"{store.carry_over(a.from_run, run, a.by)} clusters updated")
        elif a.cmd in ("check", "review", "apply-checks"):
            _review_command(store, run, cid, a)
        elif a.cmd == "progress":
            print(json.dumps(store.progress(run), indent=2))
        elif a.cmd == "export":
            if a.path.endswith((".json", ".jsonld")):
                n = store.export_web_annotations(a.path, run, a.base_uri)
            else:
                n = store.export_csv(a.path, run, a.include_removed)
            print(f"{n} occurrences written to {a.path}")
    except (AnnotationError, ImportError) as e:
        raise SystemExit(f"error: {e}")
    finally:
        store.close()


if __name__ == "__main__":
    main()