# """Transcribe word clusters with DeepSeek's vision model (one request per cluster).
#
# Why per cluster: the model sees several handwritten samples of the *same* word at once,
# which makes ambiguous letters far easier to read, and it costs one call instead of many.
# Each request also gets one line in context (the word boxed in red), because historical
# abbreviations ("Fran:co", "q.m", "Sl") are often only readable in context.
#
# Per cluster the model returns JSON:
#   diplomatic   - exactly as written, abbreviation marks kept   e.g. "Fran^co"
#   expanded     - abbreviations expanded                         e.g. "Francesco"
#   confidence   - 0..1
#   same_word    - one true/false per sample (cluster check: false = wrong member)
#   notes
#
# Unclustered words (optional, --singletons): one request per line, every word boxed and
# numbered; the model returns a reading per number.
#
# Inputs : page image
#          out_words/<stem>_words.json     (from word_clusters.py)
#          out_kraken/<stem>_kraken.json   (from kraken_lines.py)
# Output : out_transcribe/<stem>_transcriptions.json and .csv
#          --dry-run writes the images/prompts that would be sent, without calling the API.
#
# Setup  : pip install openai
#          export DEEPSEEK_API_KEY=sk-...
# Usage  : python transcribe_clusters.py page.png --dry-run
#          python transcribe_clusters.py page.png [--singletons]
#          python transcribe_clusters.py page.png --batch 0     # all clusters in ONE request
#          python transcribe_clusters.py page.png --batch 10    # 10 clusters per request
# """
# from __future__ import annotations
#
# import base64
# import csv
# import io
# import json
# import os
# import re
# from pathlib import Path
#
# import cv2
# import numpy as np
# from PIL import Image
#
# # ---------------------------------------------------------------- settings
# MODEL = "deepseek-flash"                 # DeepSeek's image-capable model
# BASE_URL = "https://api.deepseek.com"    # OpenAI-compatible endpoint
# SCRIPT_DIR = Path(__file__).resolve().parent
#
# CONTEXT = (
#     "These images come from an 18th-century Venetian register (1760s) listing foreigners "
#     "(forestieri) reported by innkeepers: names, titles (Sl = Signori, Co: = Conte, "
#     "q.m = quondam), places of origin and lodgings. The script is Italian cursive with "
#     "frequent abbreviations, superscript letters and a long s."
# )
#
# CLUSTER_PROMPT = CONTEXT + """
#
# Images 1-{n} are crops of handwritten words that a clustering algorithm grouped as the
# SAME word. The last image shows one of them in its full line, boxed in red, for context.
#
# Return ONLY a JSON object:
# {{"diplomatic": "<exactly as written; keep abbreviation marks, use ^ for superscripts, e.g. Fran^co>",
#   "expanded": "<full form with abbreviations expanded, or the same if none>",
#   "confidence": <0.0-1.0>,
#   "same_word": [<true/false for each of images 1-{n}>],
#   "notes": "<short; mention any uncertain letters>"}}
# If the samples are clearly different words, read the majority word and mark the others false.
# Use [?] for letters you cannot read; do not guess names."""
#
# LINE_PROMPT = CONTEXT + """
#
# The image is one handwritten line. Words are boxed and numbered in red.
# Return ONLY a JSON object mapping each number to its reading:
# {"1": {"diplomatic": "...", "expanded": "...", "confidence": 0.0}, ...}
# Use [?] for letters you cannot read; do not guess names."""
#
#
# BATCH_PROMPT = CONTEXT + """
#
# You will see {n} images. Each image is ONE cluster of handwritten words that a clustering
# algorithm grouped as the SAME word. Each image is titled "CLUSTER <id>". Its top part shows
# the word samples, numbered 1, 2, 3 ... above each sample; its bottom part shows the line in
# which sample 1 appears, with that word boxed in red, for context.
#
# Return ONLY one JSON object with one entry per cluster id:
# {{"<id>": {{"diplomatic": "<exactly as written; keep abbreviation marks, use ^ for superscripts>",
#            "expanded": "<full form with abbreviations expanded>",
#            "confidence": <0.0-1.0>,
#            "same_word": [<true/false for each numbered sample>],
#            "notes": "<short>"}},
#   ...}}
# Cluster ids in this request: {ids}.
# Read each cluster independently. If a cluster's samples are clearly different words, read the
# majority word and mark the others false. Use [?] for letters you cannot read; do not guess names."""
#
#
# # ---------------------------------------------------------------- paths
# def resolve(path: str | Path) -> Path:
#     """Find a file as given, or relative to this script (so PyCharm's working dir doesn't matter)."""
#     p = Path(path)
#     if p.exists():
#         return p
#     q = SCRIPT_DIR / p
#     if q.exists():
#         return q
#     raise FileNotFoundError(f"'{path}' not found (looked in {Path.cwd()} and {SCRIPT_DIR})")
#
#
# # ---------------------------------------------------------------- image helpers
# def warp_quad(rgb: np.ndarray, quad: np.ndarray, pad: int = 4) -> np.ndarray:
#     """Straightened crop of a word from its 4 corner points (TL, TR, BR, BL), slightly padded
#     so ascenders/descenders at the edge are kept."""
#     q = quad.astype(np.float32)
#     w0 = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
#     h0 = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
#     w, h = int(w0) + 2 * pad, int(h0) + 2 * pad
#     c = q.mean(0)
#     q = c + (q - c) * np.array([w / max(w0, 1), h / max(h0, 1)], np.float32)
#     dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
#     return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(q, dst), (w, h),
#                                borderMode=cv2.BORDER_REPLICATE)
#
#
# def upscale(img: np.ndarray, target_h: int = 120) -> np.ndarray:
#     """Small word crops are hard to read: enlarge to a comfortable height."""
#     s = target_h / max(1, img.shape[0])
#     return cv2.resize(img, (max(1, int(img.shape[1] * s)), target_h), interpolation=cv2.INTER_CUBIC)
#
#
# def line_context(rgb: np.ndarray, line_poly: np.ndarray, boxes: list[np.ndarray],
#                  labels: list[str] | None = None, margin: int = 25, max_w: int = 1200) -> np.ndarray:
#     """The line around a word (or several), with each word boxed (and numbered) in red."""
#     x, y, w, h = cv2.boundingRect(line_poly.astype(np.int32))
#     x0, y0 = max(0, x - margin), max(0, y - margin)
#     x1, y1 = min(rgb.shape[1], x + w + margin), min(rgb.shape[0], y + h + margin)
#     crop = rgb[y0:y1, x0:x1].copy()
#     for i, q in enumerate(boxes):
#         q = (q - [x0, y0]).astype(np.int32)
#         cv2.polylines(crop, [q], True, (220, 0, 0), 2, cv2.LINE_AA)
#         if labels:
#             cv2.putText(crop, labels[i], tuple(int(v) for v in q[0] + [0, -4]),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 0, 0), 2, cv2.LINE_AA)
#     if crop.shape[1] > max_w:
#         s = max_w / crop.shape[1]
#         crop = cv2.resize(crop, (max_w, int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
#     elif crop.shape[0] < 150:
#         crop = upscale(crop, 150)
#     return crop
#
#
# def cluster_panel(cid: str, crops: list[np.ndarray], ctx: np.ndarray, max_w: int = 1200,
#                   gap: int = 14) -> np.ndarray:
#     """One image per cluster: title, numbered samples (wrapped into rows), line context below."""
#     white = lambda h, w: np.full((h, w, 3), 255, np.uint8)
#     rows, row, row_w = [], [], 0
#     for i, c in enumerate(crops, 1):
#         c = c.copy()
#         lab = white(26, c.shape[1])
#         cv2.putText(lab, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
#         tile = np.vstack([lab, c])
#         if row and row_w + tile.shape[1] + gap > max_w:
#             rows.append(row)
#             row, row_w = [], 0
#         row.append(tile)
#         row_w += tile.shape[1] + gap
#     if row:
#         rows.append(row)
#     blocks = []
#     for r in rows:
#         h = max(t.shape[0] for t in r)
#         parts = []
#         for t in r:
#             parts += [np.vstack([t, white(h - t.shape[0], t.shape[1])]), white(h, gap)]
#         blocks.append(np.hstack(parts))
#     title = white(40, max_w)
#     cv2.putText(title, f"CLUSTER {cid}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
#     if ctx.shape[1] > max_w:
#         ctx = cv2.resize(ctx, (max_w, int(ctx.shape[0] * max_w / ctx.shape[1])), interpolation=cv2.INTER_AREA)
#     W = max([max_w] + [b.shape[1] for b in blocks])
#     pad = lambda im: np.hstack([im, white(im.shape[0], W - im.shape[1])]) if im.shape[1] < W else im
#     return np.vstack([pad(title)] + [pad(b) for b in blocks] + [white(gap, W), pad(ctx)])
#
#
# # ---------------------------------------------------------------- DeepSeek call
# def make_client():
#     from openai import OpenAI
#     key = os.environ.get("DEEPSEEK_API_KEY")
#     if not key:
#         raise RuntimeError("Set the DEEPSEEK_API_KEY environment variable first.")
#     return OpenAI(api_key=key, base_url=BASE_URL)
#
#
# def image_part(img: np.ndarray) -> dict:
#     """OpenAI-format image block (base64 PNG data URI). PNG keeps faint strokes intact;
#     detail='original' avoids the 512x512 downscale of 'low'."""
#     buf = io.BytesIO()
#     Image.fromarray(img).save(buf, format="PNG")
#     b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#     return {"type": "image_url",
#             "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "original"}}
#
#
# def ask(client, images: list[np.ndarray], prompt: str, max_tokens: int = 800) -> dict:
#     """Send images + prompt, return the parsed JSON answer (or {'error': raw_text})."""
#     content = [image_part(im) for im in images] + [{"type": "text", "text": prompt}]
#     resp = client.chat.completions.create(
#         model=MODEL,
#         max_tokens=max_tokens,
#         temperature=0,                                   # same image -> same reading
#         messages=[{"role": "user", "content": content}],
#     )
#     text = resp.choices[0].message.content or ""
#     m = re.search(r"\{.*\}", text, re.S)                 # tolerate prose around the JSON
#     try:
#         return json.loads(m.group(0)) if m else {"error": text}
#     except json.JSONDecodeError:
#         return {"error": text}
#
#
# # ---------------------------------------------------------------- pipeline
# def run(image="page.png", words_json=None, lines_json=None, out_dir="out_transcribe",
#         max_samples: int = 6, singletons: bool = False, dry_run: bool = False,
#         batch: int = 1) -> dict:
#     """batch: clusters per request. 1 = one request per cluster (most accurate),
#     0 = all clusters in a single request, N = groups of N clusters."""
#     img_path = resolve(image)
#     stem = img_path.stem
#     words_json = resolve(words_json or f"out_words/{stem}_words.json")
#     lines_json = resolve(lines_json or f"out_kraken/{stem}_kraken.json")
#
#     rgb = np.array(Image.open(img_path).convert("RGB"))
#     W = json.loads(words_json.read_text())
#     L = {l["id"]: np.array(l["polygon"], float) for l in json.loads(lines_json.read_text())["lines"]}
#     words = {w["id"]: w for w in W["words"]}
#     out = Path(out_dir) if Path(out_dir).is_absolute() else SCRIPT_DIR / out_dir
#     out.mkdir(parents=True, exist_ok=True)
#
#     client = None if dry_run else make_client()
#     results = {"model": MODEL, "clusters": {}, "words": {}}
#
#     def record(cid, members, sample, r):
#         results["clusters"][cid] = {"members": members, **r}
#         same = r.get("same_word") or []
#         for m in members:
#             k = sample.index(m) if m in sample else None
#             ok = bool(same[k]) if k is not None and k < len(same) else None   # None = not shown
#             results["words"][m] = {"cluster": cid, "diplomatic": r.get("diplomatic"),
#                                    "expanded": r.get("expanded"), "confidence": r.get("confidence"),
#                                    "cluster_check": ok}
#         print(f"cluster {cid}: {r.get('diplomatic')} -> {r.get('expanded')} "
#               f"({r.get('confidence')}){'  ERROR: ' + str(r['error'])[:80] if 'error' in r else ''}")
#
#     # prepare every cluster once
#     prepared = []
#     for cid, members in W["clusters"].items():
#         sample = members[:max_samples]
#         crops = [upscale(warp_quad(rgb, np.array(words[m]["quad"], float))) for m in sample]
#         ref = words[sample[0]]
#         ctx = line_context(rgb, L[ref["line"]], [np.array(ref["quad"], float)])
#         prepared.append((cid, members, sample, crops, ctx))
#
#     if batch == 1:
#         # 1) one request per cluster
#         for cid, members, sample, crops, ctx in prepared:
#             prompt = CLUSTER_PROMPT.format(n=len(crops))
#             if dry_run:
#                 d = out / "dry_run" / f"cluster_{int(cid):03d}"
#                 d.mkdir(parents=True, exist_ok=True)
#                 for i, c in enumerate(crops + [ctx], 1):
#                     Image.fromarray(c).save(d / f"{i:02d}.png")
#                 (d / "prompt.txt").write_text(prompt)
#                 continue
#             try:
#                 r = ask(client, crops + [ctx], prompt)
#             except Exception as e:                        # keep going if one call fails
#                 r = {"error": str(e)}
#             record(cid, members, sample, r)
#     else:
#         # 1b) several clusters per request: one panel image per cluster
#         size = len(prepared) if batch <= 0 else batch
#         for b0 in range(0, len(prepared), max(1, size)):
#             chunk = prepared[b0:b0 + size]
#             panels = [cluster_panel(cid, crops, ctx) for cid, _, _, crops, ctx in chunk]
#             ids = [cid for cid, *_ in chunk]
#             prompt = BATCH_PROMPT.format(n=len(chunk), ids=", ".join(ids))
#             if dry_run:
#                 d = out / "dry_run" / f"batch_{b0 // max(1, size):03d}"
#                 d.mkdir(parents=True, exist_ok=True)
#                 for cid, p in zip(ids, panels):
#                     Image.fromarray(p).save(d / f"cluster_{int(cid):03d}.png")
#                 (d / "prompt.txt").write_text(prompt)
#                 continue
#             try:
#                 r = ask(client, panels, prompt, max_tokens=min(8000, 250 * len(chunk) + 300))
#             except Exception as e:
#                 r = {"error": str(e)}
#             print(f"request {b0 // max(1, size) + 1}: {len(chunk)} clusters")
#             for cid, members, sample, _, _ in chunk:
#                 v = r.get(cid) if isinstance(r.get(cid), dict) else None
#                 if v is None:
#                     v = {"error": r.get("error", "cluster missing from the model's answer")}
#                 record(cid, members, sample, v)
#
#     # 2) optional: unclustered words (and words rejected from their cluster), one request per line
#     if singletons:
#         by_line: dict[str, list[dict]] = {}
#         for w in W["words"]:
#             rejected = results["words"].get(w["id"], {}).get("cluster_check") is False
#             if w["cluster"] < 0 or rejected:
#                 by_line.setdefault(w["line"], []).append(w)
#         for lid, ws in by_line.items():
#             boxes = [np.array(w["quad"], float) for w in ws]
#             ctx = line_context(rgb, L[lid], boxes, [str(i + 1) for i in range(len(ws))])
#             if dry_run:
#                 d = out / "dry_run" / f"line_{lid}"
#                 d.mkdir(parents=True, exist_ok=True)
#                 Image.fromarray(ctx).save(d / "line.png")
#                 (d / "prompt.txt").write_text(LINE_PROMPT)
#                 continue
#             try:
#                 r = ask(client, [ctx], LINE_PROMPT, max_tokens=1500)
#             except Exception as e:
#                 r = {"error": str(e)}
#             for i, w in enumerate(ws, 1):
#                 v = r.get(str(i), {}) if isinstance(r.get(str(i)), dict) else {}
#                 results["words"][w["id"]] = {"cluster": None, **v}
#             print(f"line {lid}: {len(ws)} words")
#
#     if dry_run:
#         print(f"[{stem}] dry run: request images written to {out / 'dry_run'}")
#         return results
#
#     (out / f"{stem}_transcriptions.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
#     with open(out / f"{stem}_transcriptions.csv", "w", newline="", encoding="utf-8") as f:
#         wr = csv.writer(f)
#         wr.writerow(["word_id", "line", "cluster", "diplomatic", "expanded", "confidence", "cluster_check"])
#         for wid, v in results["words"].items():
#             wr.writerow([wid, words[wid]["line"], v.get("cluster"), v.get("diplomatic"),
#                          v.get("expanded"), v.get("confidence"), v.get("cluster_check")])
#     print(f"[{stem}] transcribed {len(results['words'])} words -> {out}")
#     return results
#
#
# if __name__ == "__main__":
#     import argparse
#
#     ap = argparse.ArgumentParser(description="Transcribe word clusters with DeepSeek.")
#     ap.add_argument("image", nargs="?", default="page.png")
#     ap.add_argument("--words", default=None, help="default: out_words/<stem>_words.json")
#     ap.add_argument("--lines", default=None, help="default: out_kraken/<stem>_kraken.json")
#     ap.add_argument("--out", default="out_transcribe")
#     ap.add_argument("--max-samples", type=int, default=6, help="word images sent per cluster")
#     ap.add_argument("--singletons", action="store_true", help="also transcribe unclustered words, per line")
#     ap.add_argument("--dry-run", action="store_true", help="write the request images, do not call the API")
#     ap.add_argument("--batch", type=int, default=1,
#                     help="clusters per request: 1 = one each (default), 0 = all in ONE request, N = groups of N")
#     a = ap.parse_args()
#     run(a.image, a.words, a.lines, a.out, a.max_samples, a.singletons, a.dry_run, a.batch)

"""Transcribe word clusters with DeepSeek's vision model (one request per cluster).

Why per cluster: the model sees several handwritten samples of the *same* word at once,
which makes ambiguous letters far easier to read, and it costs one call instead of many.
Each request also gets one line in context (the word boxed in red), because historical
abbreviations ("Fran:co", "q.m", "Sl") are often only readable in context.

Per cluster the model returns JSON:
  diplomatic   - exactly as written, abbreviation marks kept   e.g. "Fran^co"
  expanded     - abbreviations expanded                         e.g. "Francesco"
  confidence   - 0..1
  same_word    - one true/false per sample (cluster check: false = wrong member)
  notes

Unclustered words (optional, --singletons): one request per line, every word boxed and
numbered; the model returns a reading per number.

Inputs : page image
         out_words/<stem>_words.json     (from word_clusters.py)
         out_kraken/<stem>_kraken.json   (from kraken_lines.py)
Output : out_transcribe/<stem>_transcriptions.json and .csv
         --dry-run writes the images/prompts that would be sent, without calling the API.

Setup  : pip install openai
         export DEEPSEEK_API_KEY=sk-...
Usage  : python transcribe_clusters.py page.png --dry-run
         python transcribe_clusters.py page.png [--singletons]
         python transcribe_clusters.py page.png --batch 0     # all clusters in ONE request
         python transcribe_clusters.py page.png --batch 10    # 10 clusters per request
"""
from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- settings
MODEL = "deepseek-flash"                 # DeepSeek's image-capable model
BASE_URL = "https://api.deepseek.com"    # OpenAI-compatible endpoint
SCRIPT_DIR = Path(__file__).resolve().parent

CONTEXT = (
    "These images come from an 18th-century Venetian register (1760s) listing foreigners "
    "(forestieri) reported by innkeepers: names, titles (Sl = Signori, Co: = Conte, "
    "q.m = quondam), places of origin and lodgings. The script is Italian cursive with "
    "frequent abbreviations, superscript letters and a long s."
)

CLUSTER_PROMPT = CONTEXT + """

Images 1-{n} are crops of handwritten words that a clustering algorithm grouped as the
SAME word. The last image shows one of them in its full line, boxed in red, for context.

Return ONLY a JSON object:
{{"diplomatic": "<exactly as written; keep abbreviation marks, use ^ for superscripts, e.g. Fran^co>",
  "expanded": "<full form with abbreviations expanded, or the same if none>",
  "confidence": <0.0-1.0>,
  "same_word": [<true/false for each of images 1-{n}>],
  "notes": "<short; mention any uncertain letters>"}}
If the samples are clearly different words, read the majority word and mark the others false.
Use [?] for letters you cannot read; do not guess names."""

LINE_PROMPT = CONTEXT + """

The image is one handwritten line. Words are boxed and numbered in red.
Return ONLY a JSON object mapping each number to its reading:
{"1": {"diplomatic": "...", "expanded": "...", "confidence": 0.0}, ...}
Use [?] for letters you cannot read; do not guess names."""


BATCH_PROMPT = CONTEXT + """

You will see {n} images. Each image is ONE cluster of handwritten words that a clustering
algorithm grouped as the SAME word. Each image is titled "CLUSTER <id>". Its top part shows
the word samples, numbered 1, 2, 3 ... above each sample; its bottom part shows the line in
which sample 1 appears, with that word boxed in red, for context.

Return ONLY one JSON object with one entry per cluster id:
{{"<id>": {{"diplomatic": "<exactly as written; keep abbreviation marks, use ^ for superscripts>",
           "expanded": "<full form with abbreviations expanded>",
           "confidence": <0.0-1.0>,
           "same_word": [<true/false for each numbered sample>],
           "notes": "<short>"}},
  ...}}
Cluster ids in this request: {ids}.
Read each cluster independently. If a cluster's samples are clearly different words, read the
majority word and mark the others false. Use [?] for letters you cannot read; do not guess names."""


# ---------------------------------------------------------------- paths
def resolve(path: str | Path) -> Path:
    """Find a file as given, or relative to this script (so PyCharm's working dir doesn't matter)."""
    p = Path(path)
    if p.exists():
        return p
    q = SCRIPT_DIR / p
    if q.exists():
        return q
    raise FileNotFoundError(f"'{path}' not found (looked in {Path.cwd()} and {SCRIPT_DIR})")


# ---------------------------------------------------------------- image helpers
def warp_quad(rgb: np.ndarray, quad: np.ndarray, pad: int = 4) -> np.ndarray:
    """Straightened crop of a word from its 4 corner points (TL, TR, BR, BL), slightly padded
    so ascenders/descenders at the edge are kept."""
    q = quad.astype(np.float32)
    w0 = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h0 = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    w, h = int(w0) + 2 * pad, int(h0) + 2 * pad
    c = q.mean(0)
    q = c + (q - c) * np.array([w / max(w0, 1), h / max(h0, 1)], np.float32)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(q, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def upscale(img: np.ndarray, target_h: int = 120) -> np.ndarray:
    """Small word crops are hard to read: enlarge to a comfortable height."""
    s = target_h / max(1, img.shape[0])
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), target_h), interpolation=cv2.INTER_CUBIC)


def line_context(rgb: np.ndarray, line_poly: np.ndarray, boxes: list[np.ndarray],
                 labels: list[str] | None = None, margin: int = 25, max_w: int = 1200) -> np.ndarray:
    """The line around a word (or several), with each word boxed (and numbered) in red."""
    x, y, w, h = cv2.boundingRect(line_poly.astype(np.int32))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(rgb.shape[1], x + w + margin), min(rgb.shape[0], y + h + margin)
    crop = rgb[y0:y1, x0:x1].copy()
    for i, q in enumerate(boxes):
        q = (q - [x0, y0]).astype(np.int32)
        cv2.polylines(crop, [q], True, (220, 0, 0), 2, cv2.LINE_AA)
        if labels:
            cv2.putText(crop, labels[i], tuple(int(v) for v in q[0] + [0, -4]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 0, 0), 2, cv2.LINE_AA)
    if crop.shape[1] > max_w:
        s = max_w / crop.shape[1]
        crop = cv2.resize(crop, (max_w, int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    elif crop.shape[0] < 150:
        crop = upscale(crop, 150)
    return crop


def cluster_panel(cid: str, crops: list[np.ndarray], ctx: np.ndarray, max_w: int = 1200,
                  gap: int = 14) -> np.ndarray:
    """One image per cluster: title, numbered samples (wrapped into rows), line context below."""
    white = lambda h, w: np.full((h, w, 3), 255, np.uint8)
    rows, row, row_w = [], [], 0
    for i, c in enumerate(crops, 1):
        c = c.copy()
        lab = white(26, c.shape[1])
        cv2.putText(lab, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)
        tile = np.vstack([lab, c])
        if row and row_w + tile.shape[1] + gap > max_w:
            rows.append(row)
            row, row_w = [], 0
        row.append(tile)
        row_w += tile.shape[1] + gap
    if row:
        rows.append(row)
    blocks = []
    for r in rows:
        h = max(t.shape[0] for t in r)
        parts = []
        for t in r:
            parts += [np.vstack([t, white(h - t.shape[0], t.shape[1])]), white(h, gap)]
        blocks.append(np.hstack(parts))
    title = white(40, max_w)
    cv2.putText(title, f"CLUSTER {cid}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    if ctx.shape[1] > max_w:
        ctx = cv2.resize(ctx, (max_w, int(ctx.shape[0] * max_w / ctx.shape[1])), interpolation=cv2.INTER_AREA)
    W = max([max_w] + [b.shape[1] for b in blocks])
    pad = lambda im: np.hstack([im, white(im.shape[0], W - im.shape[1])]) if im.shape[1] < W else im
    return np.vstack([pad(title)] + [pad(b) for b in blocks] + [white(gap, W), pad(ctx)])


# ---------------------------------------------------------------- DeepSeek call
def make_client():
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set the DEEPSEEK_API_KEY environment variable first.")
    return OpenAI(api_key=key, base_url=BASE_URL)


def image_part(img: np.ndarray) -> dict:
    """OpenAI-format image block (base64 PNG data URI). PNG keeps faint strokes intact;
    detail='original' avoids the 512x512 downscale of 'low'."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "original"}}


def ask(client, images: list[np.ndarray], prompt: str, max_tokens: int = 800) -> dict:
    """Send images + prompt, return the parsed JSON answer (or {'error': raw_text})."""
    content = [image_part(im) for im in images] + [{"type": "text", "text": prompt}]
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,                                   # same image -> same reading
        messages=[{"role": "user", "content": content}],
        # thinking is on by default and counts against max_tokens: with it on,
        # the answer can come back empty
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)                 # tolerate prose around the JSON
    try:
        return json.loads(m.group(0)) if m else {"error": text}
    except json.JSONDecodeError:
        return {"error": text}


# ---------------------------------------------------------------- pipeline
def run(image="page.png", words_json=None, lines_json=None, out_dir="out_transcribe",
        max_samples: int = 6, singletons: bool = False, dry_run: bool = False,
        batch: int = 1) -> dict:
    """batch: clusters per request. 1 = one request per cluster (most accurate),
    0 = all clusters in a single request, N = groups of N clusters."""
    img_path = resolve(image)
    stem = img_path.stem
    words_json = resolve(words_json or f"out_words/{stem}_words.json")
    lines_json = resolve(lines_json or f"out_kraken/{stem}_kraken.json")

    rgb = np.array(Image.open(img_path).convert("RGB"))
    W = json.loads(words_json.read_text())
    L = {l["id"]: np.array(l["polygon"], float) for l in json.loads(lines_json.read_text())["lines"]}
    words = {w["id"]: w for w in W["words"]}
    out = Path(out_dir) if Path(out_dir).is_absolute() else SCRIPT_DIR / out_dir
    out.mkdir(parents=True, exist_ok=True)

    client = None if dry_run else make_client()
    results = {"model": MODEL, "clusters": {}, "words": {}}

    def record(cid, members, sample, r):
        results["clusters"][cid] = {"members": members, **r}
        same = r.get("same_word") or []
        for m in members:
            k = sample.index(m) if m in sample else None
            ok = bool(same[k]) if k is not None and k < len(same) else None   # None = not shown
            results["words"][m] = {"cluster": cid, "diplomatic": r.get("diplomatic"),
                                   "expanded": r.get("expanded"), "confidence": r.get("confidence"),
                                   "cluster_check": ok}
        print(f"cluster {cid}: {r.get('diplomatic')} -> {r.get('expanded')} "
              f"({r.get('confidence')}){'  ERROR: ' + str(r['error'])[:80] if 'error' in r else ''}")

    # prepare every cluster once
    prepared = []
    for cid, members in W["clusters"].items():
        sample = members[:max_samples]
        crops = [upscale(warp_quad(rgb, np.array(words[m]["quad"], float))) for m in sample]
        ref = words[sample[0]]
        ctx = line_context(rgb, L[ref["line"]], [np.array(ref["quad"], float)])
        prepared.append((cid, members, sample, crops, ctx))

    if batch == 1:
        # 1) one request per cluster
        for cid, members, sample, crops, ctx in prepared:
            prompt = CLUSTER_PROMPT.format(n=len(crops))
            if dry_run:
                d = out / "dry_run" / f"cluster_{int(cid):03d}"
                d.mkdir(parents=True, exist_ok=True)
                for i, c in enumerate(crops + [ctx], 1):
                    Image.fromarray(c).save(d / f"{i:02d}.png")
                (d / "prompt.txt").write_text(prompt)
                continue
            try:
                r = ask(client, crops + [ctx], prompt)
            except Exception as e:                        # keep going if one call fails
                r = {"error": str(e)}
            record(cid, members, sample, r)
    else:
        # 1b) several clusters per request: one panel image per cluster
        size = len(prepared) if batch <= 0 else batch
        for b0 in range(0, len(prepared), max(1, size)):
            chunk = prepared[b0:b0 + size]
            panels = [cluster_panel(cid, crops, ctx) for cid, _, _, crops, ctx in chunk]
            ids = [cid for cid, *_ in chunk]
            prompt = BATCH_PROMPT.format(n=len(chunk), ids=", ".join(ids))
            if dry_run:
                d = out / "dry_run" / f"batch_{b0 // max(1, size):03d}"
                d.mkdir(parents=True, exist_ok=True)
                for cid, p in zip(ids, panels):
                    Image.fromarray(p).save(d / f"cluster_{int(cid):03d}.png")
                (d / "prompt.txt").write_text(prompt)
                continue
            try:
                r = ask(client, panels, prompt, max_tokens=min(8000, 250 * len(chunk) + 300))
            except Exception as e:
                r = {"error": str(e)}
            print(f"request {b0 // max(1, size) + 1}: {len(chunk)} clusters")
            for cid, members, sample, _, _ in chunk:
                v = r.get(cid) if isinstance(r.get(cid), dict) else None
                if v is None:
                    v = {"error": r.get("error", "cluster missing from the model's answer")}
                record(cid, members, sample, v)

    # 2) optional: unclustered words (and words rejected from their cluster), one request per line
    if singletons:
        by_line: dict[str, list[dict]] = {}
        for w in W["words"]:
            rejected = results["words"].get(w["id"], {}).get("cluster_check") is False
            if w["cluster"] < 0 or rejected:
                by_line.setdefault(w["line"], []).append(w)
        for lid, ws in by_line.items():
            boxes = [np.array(w["quad"], float) for w in ws]
            ctx = line_context(rgb, L[lid], boxes, [str(i + 1) for i in range(len(ws))])
            if dry_run:
                d = out / "dry_run" / f"line_{lid}"
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(ctx).save(d / "line.png")
                (d / "prompt.txt").write_text(LINE_PROMPT)
                continue
            try:
                r = ask(client, [ctx], LINE_PROMPT, max_tokens=1500)
            except Exception as e:
                r = {"error": str(e)}
            for i, w in enumerate(ws, 1):
                v = r.get(str(i), {}) if isinstance(r.get(str(i)), dict) else {}
                results["words"][w["id"]] = {"cluster": None, **v}
            print(f"line {lid}: {len(ws)} words")

    if dry_run:
        print(f"[{stem}] dry run: request images written to {out / 'dry_run'}")
        return results

    (out / f"{stem}_transcriptions.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    with open(out / f"{stem}_transcriptions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["word_id", "line", "cluster", "diplomatic", "expanded", "confidence", "cluster_check"])
        for wid, v in results["words"].items():
            wr.writerow([wid, words[wid]["line"], v.get("cluster"), v.get("diplomatic"),
                         v.get("expanded"), v.get("confidence"), v.get("cluster_check")])
    print(f"[{stem}] transcribed {len(results['words'])} words -> {out}")
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Transcribe word clusters with DeepSeek.")
    ap.add_argument("image", nargs="?", default="page.png")
    ap.add_argument("--words", default=None, help="default: out_words/<stem>_words.json")
    ap.add_argument("--lines", default=None, help="default: out_kraken/<stem>_kraken.json")
    ap.add_argument("--out", default="out_transcribe")
    ap.add_argument("--max-samples", type=int, default=6, help="word images sent per cluster")
    ap.add_argument("--singletons", action="store_true", help="also transcribe unclustered words, per line")
    ap.add_argument("--dry-run", action="store_true", help="write the request images, do not call the API")
    ap.add_argument("--batch", type=int, default=1,
                    help="clusters per request: 1 = one each (default), 0 = all in ONE request, N = groups of N")
    a = ap.parse_args()
    run(a.image, a.words, a.lines, a.out, a.max_samples, a.singletons, a.dry_run, a.batch)