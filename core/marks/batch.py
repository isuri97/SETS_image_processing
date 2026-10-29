"""Process every image in a folder: mark detection, then similar-word spotting.

For each image, in order:

  1. marks  - braces (with their member lines and notes), rules, slashes,
              dashes, circles and fillers (run.py);
  2. words  - similar-word spotting within the page (your cross_spotting.py,
              `run_spotting`). By default it runs on the page with the marks
              painted out, so braces, slashes and fillers are not segmented
              as word tokens.

Optionally, similar words are also matched *between* pages (--cross):

  next    each page with the next one, as separate runs
  all     every pair of pages, as separate runs (clusters per pair)
  folder  all pages in one run, in three stages:
            1. similar words within each document (page matched with itself)
            2. where those words occur in the other documents
            3. one set of clusters over all documents
          Each page is segmented once and feature statistics are pooled over
          the folder. The per-image word stage is then skipped, because stage
          1 already covers it.

Usage
-----
    python batch.py /path/to/images
    python batch.py /path/to/images --spotting-root /path/to/SETS_image_processing \\
        --spotting-module core.comparison_of_two_documents.experiment_2.cross_spotting
    python batch.py /path/to/images --cross next      # also page 1-2, 2-3, ...
    python batch.py /path/to/images --cross all       # also every pair, each as a separate run
    python batch.py /path/to/images --cross folder    # all pages compared together, one set of clusters
    python batch.py /path/to/images --no-words        # marks only
    python batch.py /path/to/images --skip-done       # only new or previously failed images

Run without arguments (e.g. the Run button in PyCharm) to use SETTINGS at the
bottom of this file.

Results are written into the image folder:

    results/<image>/                 marks: overlay, marks-only, text-only, brace crops, result.json,
                                     symbols.csv (one row per symbol and brace line)
    results/<image>/words/           words: within_<image>.csv/.jpg, clusters.csv/.jpg, clusters_overlay.jpg
    results/cross/<a>__<b>/          words shared by two pages (--cross next/all)
    results/folder/                  --cross folder:
        1_within/groups_<image>.csv        stage 1: word groups repeated inside each document
        1_within/<image>.jpg               the page with its groups numbered (g1, g2, ...)
        1_within/matches_<image>.csv       the within-document matches behind the groups
        2_other_documents/<image>_groups_elsewhere.csv
                                           stage 2: for each group, the other documents and
                                           tokens where it was found ("no" if nowhere else)
        2_other_documents/between/<a>__<b>.csv   all matches between two documents
        3_clusters/cluster_pages.csv       stage 3: one row per cluster: pages, scope, and the
                                           stage-1 groups it contains
        3_clusters/clusters.csv            every occurrence (page, token, box)
        3_clusters/clusters.jpg            crops of the strongest clusters
        3_clusters/pages/<image>.jpg       each page with the final cluster numbers
        symbols.csv                        the symbols.csv of all pages in one file
        page_images/<image>.png            the pages the words were spotted on, in one folder:
                                           the IMAGE_DIR for the annotation app
        page_summary.csv                   per page: groups found, how many also elsewhere
        folder_timings.csv                 time of every step
    results/timings.csv              one row per image: status, seconds per stage, counts
    results/cross_timings.csv        one row per page pair (with --cross)
    results/summary.txt              totals and average time per stage

Every clusters.csv (per page, per pair and folder-wide) gets three extra columns
linking each word occurrence to the brace lines it lies on: symbol (e.g. B2),
symbol_relation (member, borderline or note) and symbol_line (see symbol_csv.py).

Only images directly inside the folder are processed (not subfolders), so the
result images are never picked up on a second run. An image that fails is
recorded with its error message and the batch carries on.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from run import process
from symbol_csv import add_symbols, merge_symbols

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MARK_KINDS = ("rule", "slash", "dash", "circle", "filler")
COLUMNS = ["image", "status", "seconds", "marks_seconds", "words_seconds",
           "braces", "member_lines", "notes", *MARK_KINDS,
           "word_matches", "word_clusters", "word_occurrences", "word_symbol_links", "error"]
CROSS_COLUMNS = ["page_a", "page_b", "status", "seconds", "matches", "clusters", "error"]


# --------------------------------------------------------------------------
# word-spotting module
# --------------------------------------------------------------------------
# functions of your word-spotting module used by each mode
NEEDS_RUN = ("run_spotting",)
NEEDS_FOLDER = ("load_page", "match", "cluster_matches", "write_matches", "write_clusters",
                "draw_clusters", "group_palette", "CrossParams", "SpotParams", "HogPlusParams",
                "whitened_hog", "compute_hog", "project_sequences")


def load_spotter(module: str, root: Path | None, needs: tuple[str, ...]):
    """Import your word-spotting module and check it has the functions in `needs`.

    `root` is the folder from which `module` is importable (the project root
    for a dotted name such as core.comparing_all_similar_words.in_between_comparison,
    or the folder holding the file for a plain name).
    """
    if root is not None:
        sys.path.insert(0, str(Path(root).resolve()))
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise SystemExit(f"Could not import the word-spotting module '{module}'"
                         f"{f' from {root}' if root else ''}: {exc}\n"
                         "Set --spotting-root and --spotting-module, or use --no-words.")
    except Exception as exc:  # an error inside the module itself, raised while it loads
        raise SystemExit(f"The word-spotting module '{module}' was found but failed while loading:\n"
                         f"  {type(exc).__name__}: {exc}\n"
                         f"Run it on its own to see the full traceback:  python -c \"import {module}\"")
    missing = [n for n in needs if not hasattr(mod, n)]
    if missing:
        found = [n for n in dir(mod) if n.startswith("run")] or ["none"]
        raise SystemExit(f"'{module}' lacks: {', '.join(missing)}\n"
                         f"  file loaded : {getattr(mod, '__file__', '?')}\n"
                         f"  run* found  : {', '.join(found)}\n"
                         "This is probably an older cross_spotting.py; point spotting_module "
                         "at the version that defines run_spotting().")
    return mod


def make_params(mod, params):
    """CrossParams for the spotter: None -> its defaults, a dict -> defaults with
    those fields changed (e.g. {"z_threshold": 3.0}), anything else is used as is."""
    if mod is None or not isinstance(params, dict):
        return params
    unknown = [k for k in params if not hasattr(mod.CrossParams, k)]
    if unknown:
        raise SystemExit(f"Unknown CrossParams field(s): {', '.join(unknown)}")
    return mod.CrossParams(**params)


def spotting_input(image: Path, page_dir: Path, clean: bool) -> Path:
    """The image the word spotter reads.

    With `clean`, the marks-free page (text_only.png) is copied to
    words/<image stem>.png: the spotter names a page after its file stem, so
    each page must keep its own name for the cross-page clusters.
    """
    if not clean:
        return image
    src = page_dir / "text_only.png"
    dst = page_dir / "words" / f"{image.stem}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copyfile(src, dst)
    return dst


def _spot_counts(res: dict) -> dict:
    """Match and cluster counts from a run_spotting() result."""
    clusters = res.get("clusters", [])
    return {
        "word_matches": sum(len(pairs) for _, _, pairs in res.get("pairs", [])),
        "word_clusters": len(clusters),
        "word_occurrences": sum(len(c.members) for c in clusters),
    }


# --------------------------------------------------------------------------
# one image
# --------------------------------------------------------------------------
def _error_text(exc: Exception, image: Path) -> str:
    if isinstance(exc, FileNotFoundError) and image.exists():
        return "file could not be read as an image (corrupt or unsupported format)"
    return f"{type(exc).__name__}: {exc}"


def run_one(image: Path, out_root: Path, spot: Callable | None, clean: bool, params) -> dict:
    """Run both stages on one image; return its timings row.

    status is "ok", "failed" (marks stage failed, words not attempted) or
    "words_failed" (marks results are kept, the word stage failed).
    """
    row = {"image": image.name}
    start = time.perf_counter()
    try:
        counts = process(str(image), out_root)["counts"]
        row.update(status="ok", braces=counts["braces"], member_lines=sum(counts["members"]),
                   notes=counts["notes"], **{k: counts.get(k, 0) for k in MARK_KINDS})
    except Exception as exc:  # one bad image must not stop the batch
        row.update(status="failed", error=_error_text(exc, image))
        (out_root / f"{image.stem}_error.txt").write_text(traceback.format_exc())
    row["marks_seconds"] = round(time.perf_counter() - start, 2)

    if spot is not None and row["status"] == "ok":
        t = time.perf_counter()
        try:
            page_dir = out_root / image.stem
            src = spotting_input(image, page_dir, clean)
            res = spot(str(src), None, page_dir / "words", params, within=True, cross=False, verbose=False)
            row.update(_spot_counts(res))
            row["word_symbol_links"] = add_symbols(page_dir / "words" / "clusters.csv", out_root,
                                                   original=not clean) or 0
        except Exception as exc:
            row.update(status="words_failed", error=f"word spotting: {type(exc).__name__}: {exc}")
            (out_root / f"{image.stem}_words_error.txt").write_text(traceback.format_exc())
        row["words_seconds"] = round(time.perf_counter() - t, 2)

    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


# --------------------------------------------------------------------------
# page pairs
# --------------------------------------------------------------------------
def page_pairs(images: list[Path], mode: str) -> list[tuple[Path, Path]]:
    """Page pairs for the separate pair runs ("next", "all"); "folder" is run_folder()."""
    if mode == "next":
        return list(zip(images, images[1:]))
    if mode == "all":
        return list(itertools.combinations(images, 2))
    return []


def run_cross(pairs: list[tuple[Path, Path]], out_root: Path, spot: Callable, clean: bool,
              params) -> list[dict]:
    """Similar words between each pair of pages, timed per pair."""
    rows = []
    path = out_root / "cross_timings.csv"
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CROSS_COLUMNS)
        writer.writeheader()
        for n, (a, b) in enumerate(pairs, 1):
            row = {"page_a": a.name, "page_b": b.name}
            t = time.perf_counter()
            try:
                src_a = spotting_input(a, out_root / a.stem, clean)
                src_b = spotting_input(b, out_root / b.stem, clean)
                pair_dir = out_root / "cross" / f"{a.stem}__{b.stem}"
                res = spot(str(src_a), str(src_b), pair_dir, params, within=False, cross=True, verbose=False)
                add_symbols(pair_dir / "clusters.csv", out_root, original=not clean)
                c = _spot_counts(res)
                row.update(status="ok", matches=c["word_matches"], clusters=c["word_clusters"])
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            row["seconds"] = round(time.perf_counter() - t, 2)
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            print(f"[pair {n}/{len(pairs)}] {a.name} <-> {b.name}: {row['status']} in {row['seconds']} s")
    return rows


# --------------------------------------------------------------------------
# folder-wide comparison (cross="folder")
# --------------------------------------------------------------------------
FOLDER_COLUMNS = ["stage", "item", "status", "seconds", "detail", "error"]


def prepare_all(mod, pages: list, cp) -> None:
    """Feature statistics pooled over the candidates of *all* pages.

    The same steps as prepare_features() in your module, which only accepts
    two pages.
    """
    cands = [c for P in pages for c in P.cands]
    if cp.whiten:
        mod.whitened_hog(cands, mod.HogPlusParams())
    else:
        mod.compute_hog(cands, mod.SpotParams())
    if getattr(cp, "hog_dtw_weight", 0) > 0:
        mod.project_sequences(cands, mod.HogPlusParams())


def draw_page_clusters(mod, P, clusters: list):
    """One page with every cluster occurrence boxed and numbered.

    Numbers and colours are those of clusters.csv, so the same word has the
    same number on every page.
    """
    import cv2

    img = P.img.copy()
    for n, (cl, col) in enumerate(zip(clusters, mod.group_palette(len(clusters))), 1):
        for Q, i in cl.members:
            if Q is not P:
                continue
            b = Q.cands[i].box
            cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), col, 3)
            cv2.putText(img, str(n), (b.x0, max(15, b.y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return img


def _per_page(cl) -> dict[str, int]:
    """Occurrences of one cluster on each page, in page order."""
    counts: dict[str, int] = {}
    for P, _ in cl.members:
        counts[P.name] = counts.get(P.name, 0) + 1
    return counts


def _token(P, i) -> tuple[str, str]:
    """(page, token id) of candidate i: one word occurrence, whatever its trim."""
    return P.name, P.cands[i].parent


def _box(c) -> list[int]:
    return [c.box.x0, c.box.y0, c.box.x1, c.box.y1]


def write_cluster_pages(path: Path, clusters: list, token_group: dict | None = None) -> None:
    """One row per final cluster: where it occurs, and which stage-1 groups it contains.

    scope            "within page" if all occurrences are on one page,
                     "across pages" otherwise
    repeated_within  pages on which the word occurs more than once
    within_groups    the stage-1 groups (page:gN) merged into this cluster
    """
    token_group = token_group or {}
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "occurrences", "n_pages", "scope", "pages", "repeated_within",
                    "within_groups", "best_z"])
        for n, cl in enumerate(clusters, 1):
            per_page = _per_page(cl)
            groups = dict.fromkeys(token_group[_token(P, i)] for P, i in cl.members
                                   if _token(P, i) in token_group)
            w.writerow([n, len(cl.members), len(per_page),
                        "within page" if len(per_page) == 1 else "across pages",
                        "; ".join(f"{k} x{v}" for k, v in per_page.items()),
                        "; ".join(k for k, v in per_page.items() if v > 1),
                        "; ".join(groups), f"{cl.best_z:.2f}"])


def write_groups(path: Path, P, groups: list) -> None:
    """Stage 1: one row per occurrence of every word group found on page P."""
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "occurrences", "best_z", "token", "candidate", "x0", "y0", "x1", "y1"])
        for n, cl in enumerate(groups, 1):
            for Q, i in cl.members:
                c = Q.cands[i]
                w.writerow([f"{P.name}:g{n}", len(cl.members), f"{cl.best_z:.2f}", c.parent, c.tid, *_box(c)])


def write_elsewhere(path: Path, P, groups: list, hits: dict) -> int:
    """Stage 2: for every group of page P, where the word was found in the other documents.

    Returns the number of groups found in at least one other document.
    """
    found = 0
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "occurrences_on_page", "found_in_other_documents", "other_pages",
                    "best_z", "page", "token", "candidate", "x0", "y0", "x1", "y1"])
        for n, cl in enumerate(groups, 1):
            gid = f"{P.name}:g{n}"
            seen: dict = {}  # one row per occurrence on another page, keeping its best z
            for Q, j, z in hits.get(gid, []):
                key = _token(Q, j)
                if key not in seen or z > seen[key][2]:
                    seen[key] = (Q, j, z)
            per_page: dict[str, int] = {}
            for Q, _, _ in seen.values():
                per_page[Q.name] = per_page.get(Q.name, 0) + 1
            pages_txt = "; ".join(f"{k} x{v}" for k, v in per_page.items())
            if not seen:
                w.writerow([gid, len(cl.members), "no", "", "", "", "", "", "", "", "", ""])
                continue
            found += 1
            for Q, j, z in sorted(seen.values(), key=lambda t: (t[0].name, -t[2])):
                c = Q.cands[j]
                w.writerow([gid, len(cl.members), "yes", pages_txt, f"{z:.2f}", Q.name, c.parent, c.tid, *_box(c)])
    return found


def run_folder(images: list[Path], out_root: Path, mod, clean: bool, params, within_params=None,
               search: str = "within_groups", max_cluster_rows: int = 150) -> dict:
    """Similar words across a folder, in three stages.

    Stage 1  within each document: every page is matched with itself and the
             matches are grouped, giving the words that repeat on that page
             (1_within/).
    Stage 2  in the other documents: every page is matched with every other
             page, and for each stage-1 group the occurrences found on the
             other pages are listed (2_other_documents/).
    Stage 3  all documents: the stage-1 groups and their occurrences in other
             documents are linked into one set of clusters (3_clusters/).
             With search="within_groups" (default) only words that repeat on
             at least one page are followed into other documents; with
             search="all_words" every between-page match is used too, so a
             word that occurs once on each of several pages is also clustered.

    Feature statistics are pooled over all pages. `within_params` are the
    CrossParams for stage 1 (None = the same as `params`). Every step is
    timed in folder/folder_timings.csv.
    """
    import cv2

    cp = params or mod.CrossParams()
    cp_within = within_params or cp
    sp = mod.SpotParams()
    out = out_root / "folder"
    d1, d2, d3 = out / "1_within", out / "2_other_documents", out / "3_clusters"
    for d in (d1, d2 / "between", d3 / "pages"):
        d.mkdir(parents=True, exist_ok=True)
    fh = (out / "folder_timings.csv").open("w", newline="")
    log = csv.DictWriter(fh, fieldnames=FOLDER_COLUMNS)
    log.writeheader()

    def step(stage: str, item: str, fn):
        """Run fn(), time it, log it; return (result, ok)."""
        t = time.perf_counter()
        row = {"stage": stage, "item": item}
        try:
            res, detail = fn()
            row.update(status="ok", detail=detail)
            ok = True
        except Exception as exc:
            res, ok = None, False
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            (out / f"{stage}_{item.replace(' ', '_')}_error.txt").write_text(traceback.format_exc())
        row["seconds"] = round(time.perf_counter() - t, 2)
        log.writerow(row)
        fh.flush()
        note = f" ({row['detail']})" if ok and row["detail"] else ""
        print(f"  [folder] {stage:<9} {item}: {row['status']} in {row['seconds']} s{note}")
        return res, ok

    t0 = time.perf_counter()
    # ---- preparation: segment each page once, pool the feature statistics
    pages = []
    for image in images:
        def seg(image=image):
            P = mod.load_page(str(spotting_input(image, out_root / image.stem, clean)), cp, sp)
            return P, f"{len(P.cands)} candidates"
        P, ok = step("segment", image.name, seg)
        if ok:
            pages.append(P)
    if not pages:
        fh.close()
        print("  [folder] no page could be segmented; folder comparison skipped")
        return {"pages": 0}
    step("features", f"{len(pages)} pages", lambda: (prepare_all(mod, pages, cp),
                                                    f"{sum(len(P.cands) for P in pages)} candidates pooled"))

    # ---- stage 1: similar words within each document
    print("Stage 1: similar words within each document")
    within, groups, token_group = {}, {}, {}
    for P in pages:
        def stage1(P=P):
            pairs = mod.match(P, P, cp_within)
            g = mod.cluster_matches([(P, P, pairs)])
            return (pairs, g), f"{len(pairs)} matches, {len(g)} word groups"
        res, ok = step("1_within", P.name, stage1)
        pairs, g = res if ok else ([], [])
        within[P.name], groups[P.name] = pairs, g
        for n, cl in enumerate(g, 1):
            for Q, i in cl.members:
                token_group[_token(Q, i)] = f"{P.name}:g{n}"
        mod.write_matches(d1 / f"matches_{P.name}.csv", P, P, pairs)
        write_groups(d1 / f"groups_{P.name}.csv", P, g)
        cv2.imwrite(str(d1 / f"{P.name}.jpg"), draw_page_clusters(mod, P, g), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ---- stage 2: do those words occur in the other documents?
    print("Stage 2: the within-document groups in the other documents")
    between, hits = [], {}  # hits: group id -> [(page, candidate, z), ...]
    for A, B in itertools.combinations(pages, 2):
        pairs, ok = step("2_between", f"{A.name}__{B.name}",
                         lambda A=A, B=B: (lambda m: (m, f"{len(m)} matches"))(mod.match(A, B, cp)))
        if not ok:
            continue
        between.append((A, B, pairs))
        if pairs:
            mod.write_matches(d2 / "between" / f"{A.name}__{B.name}.csv", A, B, pairs)
        for m in pairs:
            ga, gb = token_group.get(_token(A, m["i"])), token_group.get(_token(B, m["j"]))
            if ga:
                hits.setdefault(ga, []).append((B, m["j"], m["z"]))
            if gb:
                hits.setdefault(gb, []).append((A, m["i"], m["z"]))
    found = {P.name: write_elsewhere(d2 / f"{P.name}_groups_elsewhere.csv", P, groups[P.name], hits)
             for P in pages}
    if not token_group and search == "within_groups":
        print("  WARNING: stage 1 found no repeated words in any document, so there is nothing to\n"
              "  follow into the other documents. Lower the stage-1 threshold (within_params=\n"
              "  dict(z_threshold=3.0)) or set search=\"all_words\" to cluster all matched words.")

    # ---- stage 3: one set of clusters over all documents
    print("Stage 3: clustering the similar words of all documents")
    if search == "within_groups":
        linked = [(A, B, [m for m in pairs if _token(A, m["i"]) in token_group
                          or _token(B, m["j"]) in token_group]) for A, B, pairs in between]
    else:
        linked = between
    links = [(P, P, within[P.name]) for P in pages] + linked
    clusters, _ = step("3_cluster", "all pages",
                       lambda: (lambda c: (c, f"{len(c)} clusters"))(mod.cluster_matches(links)))
    clusters = clusters or []

    def write():
        mod.write_clusters(d3 / "clusters.csv", clusters)
        linked = add_symbols(d3 / "clusters.csv", out_root, original=not clean)
        merge_symbols([P.name for P in pages], out_root, out / "symbols.csv", original=not clean)
        # the annotation app crops words from these, so they must be the images the boxes refer to
        (out / "page_images").mkdir(exist_ok=True)
        for image in images:
            src = spotting_input(image, out_root / image.stem, clean)
            shutil.copyfile(src, out / "page_images" / src.name)
        write_cluster_pages(d3 / "cluster_pages.csv", clusters, token_group)
        # the strongest clusters only: a picture of hundreds would exceed the JPEG size limit
        cv2.imwrite(str(d3 / "clusters.jpg"), mod.draw_clusters(clusters[:max_cluster_rows]))
        for P in pages:
            cv2.imwrite(str(d3 / "pages" / f"{P.name}.jpg"), draw_page_clusters(mod, P, clusters),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        return None, f"written; {linked or 0} word occurrences linked to a brace line"
    step("write", "outputs", write)

    # ---- per page summary
    rows = []
    for P in pages:
        rows.append({"page": P.name, "within_matches": len(within[P.name]),
                     "within_groups": len(groups[P.name]),
                     "groups_found_in_other_documents": found[P.name],
                     "groups_only_in_this_document": len(groups[P.name]) - found[P.name],
                     "final_clusters_on_page": sum(P.name in _per_page(c) for c in clusters)})
        print(f"  {P.name}: {rows[-1]['within_groups']} word groups within the document, "
              f"{found[P.name]} of them also in other documents")
    with (out / "page_summary.csv").open("w", newline="") as f2:
        w = csv.DictWriter(f2, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fh.close()

    spans = [_per_page(c) for c in clusters]
    return {"pages": len(pages), "within": sum(len(v) for v in within.values()),
            "groups": sum(len(v) for v in groups.values()), "found": sum(found.values()),
            "between": sum(len(p) for _, _, p in between), "clusters": len(clusters),
            "multi_page": sum(len(x) > 1 for x in spans), "single_page": sum(len(x) == 1 for x in spans),
            "search": search, "seconds": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _stage_line(label: str, values: list) -> list[str]:
    vals = [v for v in map(_num, values) if v is not None]
    if not vals:
        return []
    return [f"{label:<17}: average {sum(vals) / len(vals):.1f} s, "
            f"fastest {min(vals):.1f} s, slowest {max(vals):.1f} s (n={len(vals)})"]


def write_summary(rows: list[dict], cross: list[dict], path: Path, total: float,
                  folder: dict | None = None) -> str:
    """Write and return a short plain-text summary of the batch."""
    ok = [r for r in rows if r["status"] == "ok"]
    lines = [
        f"images processed : {len(rows)}",
        f"succeeded        : {len(ok)}",
        f"marks failed     : {sum(r['status'] == 'failed' for r in rows)}",
        f"words failed     : {sum(r['status'] == 'words_failed' for r in rows)}",
        f"time of this run : {total:.1f} s",
    ]
    lines += _stage_line("per image", [r["seconds"] for r in ok])
    lines += _stage_line("  marks stage", [r.get("marks_seconds") for r in ok])
    lines += _stage_line("  words stage", [r.get("words_seconds") for r in ok])
    if cross:
        good = [r for r in cross if r["status"] == "ok"]
        lines += [f"page pairs       : {len(cross)} ({len(cross) - len(good)} failed)"]
        lines += _stage_line("per page pair", [r["seconds"] for r in good])
    if folder and "clusters" in folder:
        lines += [
            f"folder comparison: {folder['pages']} pages in {folder['seconds']:.1f} s",
            f"  stage 1        : {folder['groups']} word groups within the documents "
            f"({folder['within']} within-document matches)",
            f"  stage 2        : {folder['found']} of those groups also found in other documents "
            f"({folder['between']} between-document matches)",
            f"  stage 3        : {folder['clusters']} clusters ({folder['multi_page']} across documents, "
            f"{folder['single_page']} in one document only; search={folder['search']})",
            "  per page: folder/page_summary.csv",
        ]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def find_images(folder: Path) -> list[Path]:
    """Image files directly inside `folder`, in name order."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)


def is_done(image: Path, out_root: Path, words: bool) -> bool:
    page = out_root / image.stem
    return (page / "result.json").exists() and (not words or (page / "words" / "clusters.csv").exists())


def run_batch(folder: str | Path, words: bool = True, spotting_module: str = "cross_spotting",
              spotting_root: str | Path | None = None, spotting_params=None,
              clean: bool = True, cross: str = "none", skip_done: bool = False,
              within_params=None, search: str = "within_groups") -> None:
    """Process a folder of images; see the module docstring for the outputs.

    within_params  CrossParams for stage 1 of cross="folder" (similar words
                   within a document); None = the same as spotting_params
    search         cross="folder", stage 3: "within_groups" follows only the
                   words grouped in stage 1 into other documents;
                   "all_words" also clusters words that occur once per page
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    all_images = find_images(folder)
    if not all_images:
        raise SystemExit(f"No images ({', '.join(sorted(IMAGE_TYPES))}) found in {folder}")
    if cross not in ("none", "next", "all", "folder"):
        raise SystemExit(f"cross must be none, next, all or folder, not {cross!r}")
    # "folder" compares every page within itself too, so the per-image word stage is not repeated
    per_image_words = words and cross != "folder"
    needs = (NEEDS_RUN if per_image_words or cross in ("next", "all") else ()) + \
            (NEEDS_FOLDER if cross == "folder" else ())
    mod = load_spotter(spotting_module, spotting_root, needs) if needs else None
    spot = mod.run_spotting if mod is not None and hasattr(mod, "run_spotting") else None
    if search not in ("within_groups", "all_words"):
        raise SystemExit(f"search must be within_groups or all_words, not {search!r}")
    spotting_params = make_params(mod, spotting_params)
    within_params = make_params(mod, within_params)

    out_root = folder / "results"
    out_root.mkdir(exist_ok=True)
    images = [p for p in all_images if not (skip_done and is_done(p, out_root, per_image_words))]

    csv_path = out_root / "timings.csv"
    earlier = []  # with --skip-done, keep the rows of earlier runs and add to them
    if skip_done and csv_path.exists():
        with csv_path.open(newline="") as fh:
            earlier = list(csv.DictReader(fh))

    rows, t0 = [], time.perf_counter()
    with csv_path.open("a" if earlier else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        if not earlier:
            writer.writeheader()
        for i, image in enumerate(images, 1):
            row = run_one(image, out_root, spot if per_image_words else None, clean, spotting_params)
            writer.writerow(row)
            fh.flush()  # the CSV stays usable if the batch is interrupted
            rows.append(row)
            extra = f" (marks {row['marks_seconds']} s, words {row['words_seconds']} s)" \
                if "words_seconds" in row else ""
            print(f"[{i}/{len(all_images) if not skip_done else len(images)}] {image.name}: "
                  f"{row['status']} in {row['seconds']} s{extra}")

    # one row per image: a retried image replaces its earlier row
    done = {r["image"] for r in rows}
    merged = sorted([r for r in earlier if r["image"] not in done] + rows, key=lambda r: r["image"])
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    # pairs use every page whose marks stage succeeded, not only this run's
    good = {r["image"] for r in merged if r["status"] != "failed"}
    usable = [p for p in all_images if p.name in good]
    pairs = page_pairs(usable, cross)
    cross_rows = run_cross(pairs, out_root, spot, clean, spotting_params) if pairs else []
    folder_res = None
    if cross == "folder":
        print(f"Comparing the tokens of {len(usable)} pages with each other "
              f"({len(usable) * (len(usable) - 1) // 2} page pairs) ...")
        folder_res = run_folder(usable, out_root, mod, clean, spotting_params, within_params, search)

    print(write_summary(merged, cross_rows, out_root / "summary.txt", time.perf_counter() - t0, folder_res))
    print(f"Results in {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder containing the page images")
    ap.add_argument("--no-words", dest="words", action="store_false",
                    help="marks only; skip similar-word spotting")
    ap.add_argument("--spotting-module", default="cross_spotting",
                    help="import name of your word-spotting module (default: cross_spotting)")
    ap.add_argument("--spotting-root", type=Path, default=None,
                    help="folder from which --spotting-module is importable")
    ap.add_argument("--spot-on", choices=("clean", "original"), default="clean",
                    help="spot words on the page with marks painted out (default) or on the original")
    ap.add_argument("--cross", choices=("none", "next", "all", "folder"), default="none",
                    help="also match words between pages: consecutive pages, every pair as "
                         "separate runs, or all pages together with one set of clusters (folder)")
    ap.add_argument("--search", choices=("within_groups", "all_words"), default="within_groups",
                    help="--cross folder: follow only words that repeat within a document into the "
                         "other documents (default), or cluster all matched words")
    ap.add_argument("--within-z", type=float, default=None,
                    help="--cross folder: z_threshold for stage 1 only (within a document)")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip images already processed (failed images are retried)")
    a = ap.parse_args()
    within = {"z_threshold": a.within_z} if a.within_z is not None else None
    run_batch(a.folder, a.words, a.spotting_module, a.spotting_root, None,
              a.spot_on == "clean", a.cross, a.skip_done, within, a.search)


# --------------------------------------------------------------------------
# Run settings - used when the script is started without arguments
# (e.g. the Run button in PyCharm).
# --------------------------------------------------------------------------
SETTINGS = dict(
    folder="/home/isuri/PycharmProjects/SETS_image_processing/data2",   # image folder
    words=True,                                    # similar-word spotting per page
    spotting_root="/home/isuri/PycharmProjects/SETS_image_processing",      # project root
    spotting_module="core.comparing_all_similar_words.in_between_comparison",
    # None = CrossParams() defaults of your module; a dict changes single fields, e.g.
    # dict(z_threshold=3.0, top_k_within=8) for more within-page matches (and more false ones)
    spotting_params=None,
    # stage 1 only (similar words within a document); None = same as spotting_params,
    # e.g. dict(z_threshold=3.0) finds more repeats inside a page
    within_params=None,
    search="within_groups",                        # or "all_words" (see run_folder)
    clean=True,                                    # spot on the marks-free page
    cross="folder",                                # "none", "next", "all" or "folder"
    skip_done=False,
)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_batch(**SETTINGS)