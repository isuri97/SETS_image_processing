"""
VLM recall for the word-spotting annotator (vlm_recall.py)

The spotting (batch_processing.run_spotting) is tuned for precision: a token
only joins a cluster if it survives the z threshold, the margin to the
query's best, reciprocal top-k, the size shortlist and cluster_min_z. True
occurrences written by another hand, abbreviated, or of a frequent word
(more occurrences than top_k) are lost on the way.

This module recovers them in two cheap-then-expensive stages:

  1. candidates (no VLM)  every segmented token that NO cluster holds is
                          scored against every open cluster with the same
                          dense measure as spotting_app.check(): mean
                          whitened-HOG cosine to the cluster's members, as a
                          z-score against random token pairs. Loose
                          thresholds: recall matters here, not precision.
  2. verification (VLM)   for each candidate, the VLM sees a few members of
                          the cluster and the candidate, and answers whether
                          the candidate writes the same word (abbreviations
                          and other hands allowed). Only "yes" is added.

Accepted tokens are added with Annotation.add_word (origin "vlm", annotator
"vlm"), so they are logged in the history, shown in the cluster view and
re-checked by the fit check like any other word. Words the annotator has
already removed are never proposed again.

New clusters (words that recur but that the spotting never clustered):

  4. proposal (no VLM)    unclustered tokens are linked to their mutual
                          nearest neighbours (same whitened-HOG z, looser
                          threshold) and grouped with Louvain;
  5. grouping (VLM)       each group is shown to the VLM as numbered crops;
                          it splits it into sets that write the same word and
                          reads each set. A set whose reading matches an
                          existing cluster's name is offered for that cluster
                          instead of a new one.
  Proposals are only created when the annotator accepts them in the review.

Also here:
  - verify_word / first_match: the same check for a single word image, e.g.
    a Kraken word that no cluster covers (word_panel in spotting_app.py);
  - read_each: one reading PER word image (instead of one reading for the
    whole cluster), so impure clusters show up and readings can be compared.

Wiring (spotting_app.py, in main(), after `checks = check(...)` and
`deepseek_settings()`):

    import vlm_recall as vr
    vr.recall_panel(stem, get_ann, T, mu, sd, words, rgb)

The model is chosen in the panel: Qwen through Alibaba Cloud Model Studio
(OpenAI-compatible API), Qwen on a local GPU (transformers), or DeepSeek
(deepseek-flash, through transcribe_clusters.py). Any function
ask(images, prompt) -> dict | list | None can be passed instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections import Counter
from typing import Callable, Iterable

import cv2
import numpy as np

try:                                   # the app's DeepSeek helpers (optional)
    import transcribe_clusters as tc
except ImportError:                    # pragma: no cover
    tc = None

OPEN = ("pending", "labelled")
Ask = Callable[[list[np.ndarray], str], "dict | list | None"]


# =========================================================================== #
# 1. Candidates from the spotting's own features (no VLM)
# =========================================================================== #
@dataclass
class Candidate:
    wid: str          # token id "<page>:<token>" (same ids as clusters.csv)
    cid: str          # cluster it may belong to
    z: float          # mean-similarity z to that cluster's members
    row: int          # row of the token in the token table T


def cluster_sums(T: dict, groups: dict[str, list[str]]):
    """Per open cluster: summed member features, size, median width (line
    pitches). Members missing from T are ignored."""
    cids, sums, sizes, widths = [], [], [], []
    for cid, members in groups.items():
        rows = [T["pos"][w] for w in members if w in T["pos"]]
        if rows:
            cids.append(cid)
            sums.append(T["F"][rows].sum(0))
            sizes.append(len(rows))
            widths.append(float(np.median(T["width"][rows])))
    d = T["F"].shape[1]
    return (cids, np.asarray(sums, np.float32).reshape(-1, d),
            np.asarray(sizes, float), np.asarray(widths, float))


def recall_candidates(ann, T: dict, mu: float, sd: float, z_min: float = 1.0,
                      width_ratio: float = 1.6, min_width: float = 0.6,
                      per_token: int = 2, min_members: int = 2) -> list[Candidate]:
    """Tokens no cluster holds, paired with the open clusters they resemble.

    z_min        min mean-similarity z (fit check default is 1.5; lower here
                 because the VLM checks every candidate)
    width_ratio  max ratio of token width to the cluster's median width
                 (looser than the spotting's 1.3: abbreviations are shorter)
    min_width    tokens narrower than this (line pitches) are skipped
    per_token    at most this many clusters per token, best first
    min_members  only clusters with at least this many words are targets
    Returns the candidates, best z first."""
    known = set(ann.data["words"])          # clustered, removed or added by hand
    groups = {cid: ann.members(cid) for cid, c in ann.clusters.items()
              if c["status"] in OPEN}
    groups = {cid: m for cid, m in groups.items() if len(m) >= min_members}
    if not groups:
        return []
    cids, sums, sizes, widths = cluster_sums(T, groups)
    if not cids:
        return []
    rows = [i for i, w in enumerate(T["ids"])
            if w not in known and T["width"][i] >= min_width]
    if not rows:
        return []
    F = T["F"][rows]
    z = ((F @ sums.T) / sizes - mu) / sd                       # tokens x clusters
    w_tok = np.maximum(T["width"][rows], 1e-3)[:, None]
    w_cl = np.maximum(widths, 1e-3)[None, :]
    z[np.maximum(w_tok / w_cl, w_cl / w_tok) > width_ratio] = -np.inf
    out = []
    for k, i in enumerate(rows):
        for j in np.argsort(-z[k])[:per_token]:
            if np.isfinite(z[k, j]) and z[k, j] >= z_min:
                out.append(Candidate(T["ids"][i], cids[j], float(z[k, j]), i))
    return sorted(out, key=lambda c: -c.z)


# =========================================================================== #
# 2. VLM verification
# =========================================================================== #
VERIFY_PROMPT = """You see handwritten words from an 18th-century Venetian register
(Italian, cursive, heavily abbreviated).
Images 1 to {n} are confirmed occurrences of the same word{label}.
The LAST image is a candidate.

Is the candidate the same word? Count it as the same if it is the same word
written by another hand, abbreviated (e.g. sud.to / suddetto, S.ta / Santa,
Gio: / Giovanni) or with a different ending of the same word (detto / detti).
Count it as different if it is another word, only part of the word, or the
word together with a neighbouring word.

Reply with JSON only:
{{"same": true or false, "reading": "<the candidate as written>", "confidence": <0-1>}}"""

READ_EACH_PROMPT = """You see {n} handwritten word images from an 18th-century Venetian
register (Italian, cursive, abbreviated), numbered 1 to {n} in order.
Read EACH image on its own; do not assume they are the same word. An image
may be numbered in its top-left corner. If an image is not one clear word
(several lines, a fragment, a blot), give "" for it.
Reply with JSON only, one object per image, in order:
[{{"diplomatic": "<as written>", "expanded": "<abbreviations expanded>"}}, ...]"""


def parse_json(text: str):
    """First JSON object or array in a model reply (None if there is none)."""
    text = re.sub(r"```[a-z]*", "", text or "")
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


def deepseek_ask_json(images: list[np.ndarray], prompt: str):
    """The app's DeepSeek client (transcribe_clusters.py), thinking off,
    temperature 0; returns the parsed JSON reply or None."""
    if tc is None:
        raise ImportError("transcribe_clusters.py is not importable")
    client = tc.make_client()
    content = [tc.image_part(im) for im in images] + [{"type": "text", "text": prompt}]
    resp = client.chat.completions.create(
        model=tc.MODEL, temperature=0, max_tokens=400,
        messages=[{"role": "system", "content": "You compare and transcribe handwritten "
                                                "words. Reply with JSON only."},
                  {"role": "user", "content": content}],
        extra_body={"thinking": {"type": "disabled"}})
    return parse_json(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# Qwen backends (same signature: ask(images, prompt) -> parsed JSON or None)
# --------------------------------------------------------------------------- #
QWEN_API_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"   # Singapore
QWEN_API_MODEL = "qwen3-vl-plus"
QWEN_LOCAL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def data_url(img: np.ndarray) -> str:
    """RGB array -> base64 PNG data URL (OpenAI-style image_url)."""
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def openai_compatible_ask(base_url: str, model: str, api_key: str | None = None,
                          key_env: str = "DASHSCOPE_API_KEY") -> Ask:
    """An ask() for any OpenAI-compatible vision endpoint (Qwen on Alibaba
    Cloud Model Studio by default; also OpenRouter, a local vLLM server ...)."""
    import os
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get(key_env), base_url=base_url)

    def ask(images: list[np.ndarray], prompt: str):
        content = [{"type": "image_url", "image_url": {"url": data_url(im)}} for im in images]
        content.append({"type": "text", "text": prompt})
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=400,
            messages=[{"role": "system", "content": "You compare and transcribe handwritten "
                                                    "words. Reply with JSON only."},
                      {"role": "user", "content": content}])
        return parse_json(resp.choices[0].message.content)
    return ask


class QwenLocal:
    """Qwen-VL on your own GPU (Hugging Face transformers). Loaded on first
    use; ~16 GB GPU memory for the 7B model in bf16, ~8 GB for the 3B.
    Qwen3-VL models work the same way with a recent transformers version."""

    def __init__(self, model_id: str = QWEN_LOCAL_MODEL, max_new_tokens: int = 300):
        self.model_id, self.max_new_tokens = model_id, max_new_tokens
        self.model = self.processor = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        # word crops are small: cap the pixels per image to keep prompts short
        self.processor = AutoProcessor.from_pretrained(self.model_id, max_pixels=512 * 28 * 28)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto")

    def __call__(self, images: list[np.ndarray], prompt: str):
        from PIL import Image
        if self.model is None:
            self._load()
        content = [{"type": "image", "image": Image.fromarray(im)} for im in images]
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]
        return parse_json(text)


def crop_box(img: np.ndarray, box, h: int = 80) -> np.ndarray:
    """A word's box with a small margin, scaled to height h (as word_image)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = img.shape[:2]
    m = max(2, (y1 - y0) // 10)
    crop = img[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    if tc is not None and hasattr(tc, "upscale"):
        return tc.upscale(crop, h)
    s = h / max(crop.shape[0], 1)
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)


def verify_word(candidate: np.ndarray, exemplars: list[np.ndarray], ask: Ask,
                label: str | None = None, min_conf: float = 0.5) -> tuple[bool, dict]:
    """Does `candidate` write the same word as the `exemplars`?
    Returns (accepted, the model's reply). Unparseable replies are 'no'."""
    hint = f' (read as "{label}")' if label else ""
    reply = ask(exemplars + [candidate], VERIFY_PROMPT.format(n=len(exemplars), label=hint))
    if not isinstance(reply, dict):
        return False, {"error": "no JSON reply"}
    conf = reply.get("confidence")
    ok = reply.get("same") is True and (not isinstance(conf, (int, float)) or conf >= min_conf)
    return ok, reply


def exemplar_ids(ann, cid: str, T: dict, n: int) -> list[str]:
    """Up to n members of a cluster for the prompt: the ones closest to the
    cluster's mean feature (typical forms, not outliers)."""
    mem = ann.members(cid)
    rows = [(w, T["pos"][w]) for w in mem if w in T["pos"]]
    if len(rows) <= n:
        return [w for w, _ in rows] or mem[:n]
    F = T["F"][[r for _, r in rows]]
    c = F.mean(0)
    order = np.argsort(-(F @ c))
    return [rows[k][0] for k in order[:n]]


def first_match(candidate: np.ndarray, ranked: Iterable[str],
                exemplars: Callable[[str], list[np.ndarray]],
                labels: Callable[[str], str | None], ask: Ask,
                max_tries: int = 3) -> tuple[str | None, dict]:
    """Try the clusters in `ranked` order; return the first the VLM accepts
    (None if none of the first `max_tries` does) and its reply. For single
    words, e.g. a Kraken word that no cluster covers."""
    reply: dict = {}
    for cid in list(ranked)[:max_tries]:
        ok, reply = verify_word(candidate, exemplars(cid), ask, labels(cid))
        if ok:
            return cid, reply
    return None, reply


def read_each(images: list[np.ndarray], ask: Ask) -> list[dict]:
    """One reading per image ({"diplomatic", "expanded"}; empty dicts where the
    model gave none). Use instead of one reading for the whole cluster."""
    reply = ask(images, READ_EACH_PROMPT.format(n=len(images)))
    if not isinstance(reply, list):
        return [{} for _ in images]
    out = [r if isinstance(r, dict) else {} for r in reply[:len(images)]]
    return out + [{} for _ in range(len(images) - len(out))]


# =========================================================================== #
# 3. Run and add
# =========================================================================== #
def token_word(T: dict, row: int) -> dict:
    """A token of the table as an annotation word (segmented-page coordinates,
    the same id the spotting would give it)."""
    x0, y0, x1, y1 = (int(v) for v in T["box"][row])
    return dict(id=T["ids"][row], page=T["page"][row], token=T["token"][row],
                line=T["token"][row], candidate=T["tid"][row], box=[x0, y0, x1, y1],
                quad=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], best_z=None, origin="vlm")


def label_of(c: dict) -> str | None:
    """A cluster's reading for the prompt: the annotator's value, else the
    automatic name (None if neither)."""
    return c.get("value") or c.get("auto_name")


def run_recall(ann, T: dict, mu: float, sd: float, words: dict, rgb: dict, ask: Ask,
               z_min: float = 1.0, width_ratio: float = 1.6, n_exemplars: int = 4,
               max_calls: int = 100, progress: Callable[[float, str], None] | None = None,
               dry_run: bool = False) -> dict:
    """Stages 1-3 on one annotation (modified in place, not saved).

    Each token is tried against its candidate clusters best-z first and
    added to the first one the VLM accepts; at most `max_calls` VLM calls.
    Returns counts and a log row per call (for review / evaluation)."""
    cands = recall_candidates(ann, T, mu, sd, z_min, width_ratio)
    done, log, added, calls = set(), [], 0, 0
    img_cache: dict[str, np.ndarray] = {}

    def image_of(wid: str, word: dict | None = None) -> np.ndarray:
        if wid not in img_cache:
            w = word or words[wid]
            img_cache[wid] = crop_box(rgb[w.get("img", w["page"])], w["box"])
        return img_cache[wid]

    for c in cands:
        if calls >= max_calls:
            break
        if c.wid in done:
            continue
        word = token_word(T, c.row)
        ex_ids = [w for w in exemplar_ids(ann, c.cid, T, n_exemplars) if w in words]
        if not ex_ids:
            continue
        calls += 1
        if progress:
            progress(calls / max_calls, f"{c.wid} → #{ann.clusters[c.cid]['number']}")
        ok, reply = verify_word(image_of(c.wid, word), [image_of(w) for w in ex_ids], ask,
                                label_of(ann.clusters[c.cid]))
        log.append(dict(wid=c.wid, cluster=c.cid, number=ann.clusters[c.cid]["number"],
                        label=label_of(ann.clusters[c.cid]), z=round(c.z, 2), accepted=ok,
                        reading=reply.get("reading"), confidence=reply.get("confidence"),
                        exemplars=ex_ids, word=word))   # word: geometry, for the review
        if ok:
            done.add(c.wid)
            if not dry_run:
                ann.add_word(word, c.cid, "vlm")
                ann.data["readings"][c.wid] = dict(
                    diplomatic=reply.get("reading"), expanded=reply.get("reading"),
                    confidence=reply.get("confidence"), cluster_check=True)
                words[c.wid] = word
            added += 1
    return dict(candidates=len(cands), calls=calls, added=added, log=log)



# =========================================================================== #
# 3b. New clusters among the tokens no cluster holds
# =========================================================================== #
GROUP_PROMPT = """You see {n} handwritten word images from an 18th-century Venetian
register (Italian, cursive, abbreviated). Each image has its number written
in the top-left corner.
Group the images that show the SAME word (another hand, an abbreviation or a
different ending of the same word still count as the same word). Leave out
images that match no other image.
Reply with JSON only:
{{"groups": [{{"members": [<image numbers>], "reading": "<the word, abbreviations expanded>"}}]}}"""


def numbered(img: np.ndarray, k: int) -> np.ndarray:
    """The crop with a white band on top carrying its number (for the VLM)."""
    band = np.full((22, max(img.shape[1], 40), 3), 255, np.uint8)
    cv2.putText(band, str(k), (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2)
    body = np.full((img.shape[0], band.shape[1], 3), 255, np.uint8)
    body[:, :img.shape[1]] = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return np.vstack([band, body])


def norm_text(t: str | None) -> str:
    """Lower case, no accents or punctuation (for comparing readings)."""
    import unicodedata
    t = unicodedata.normalize("NFKD", t or "")
    return re.sub(r"[^a-z]", "", "".join(c for c in t if not unicodedata.combining(c)).lower())


def page_heights(T: dict) -> dict[str, float]:
    """Median token box height per page (px): the height of one line of writing."""
    if "_h" not in T:
        h = T["box"][:, 3] - T["box"][:, 1]
        pages = np.asarray(T["page"])
        T["_h"] = {p: float(np.median(h[pages == p])) for p in set(T["page"])}
    return T["_h"]


def too_tall(T: dict, i: int, max_height: float) -> bool:
    """True for a token box much taller than a line (a crop spanning several
    lines, a bracket, a flourish): such crops make misleading groups."""
    return (T["box"][i][3] - T["box"][i][1]) > max_height * page_heights(T)[T["page"][i]]


def propose_groups(ann, T: dict, mu: float, sd: float, z_min: float = 2.0, k: int = 5,
                   width_ratio: float = 1.4, min_width: float = 1.0, max_size: int = 12,
                   max_height: float = 1.8, chunk: int = 2000) -> list[list[int]]:
    """Groups of unclustered tokens (rows of T) that look alike: mutual
    k-nearest neighbours with z >= z_min and comparable widths, split with
    Louvain. Groups larger than max_size keep their best-connected members.
    Returns lists of rows, largest first."""
    import networkx as nx
    known = set(ann.data["words"])
    pool = np.array([i for i, w in enumerate(T["ids"])
                     if w not in known and T["width"][i] >= min_width
                     and not too_tall(T, i, max_height)], int)
    if len(pool) < 2:
        return []
    F, W = T["F"][pool], np.maximum(T["width"][pool], 1e-3)
    k = min(k, len(pool) - 1)
    nn = np.empty((len(pool), k), int)
    nz = np.empty((len(pool), k), np.float32)
    for a in range(0, len(pool), chunk):                 # blockwise: no n x n matrix
        z = (F[a:a + chunk] @ F.T - mu) / sd
        r = np.maximum(W[a:a + chunk, None] / W[None, :], W[None, :] / W[a:a + chunk, None])
        z[r > width_ratio] = -np.inf
        z[np.arange(len(z)), np.arange(a, a + len(z))] = -np.inf
        idx = np.argpartition(-z, k - 1, axis=1)[:, :k]
        nn[a:a + chunk], nz[a:a + chunk] = idx, np.take_along_axis(z, idx, 1)
    G = nx.Graph()
    for i in range(len(pool)):
        for j, zz in zip(nn[i], nz[i]):
            if zz >= z_min and i in nn[j]:
                G.add_edge(i, int(j), weight=float(zz))
    groups = []
    for comm in nx.community.louvain_communities(G, weight="weight", seed=0):
        if len(comm) < 2:
            continue
        sub = G.subgraph(comm)
        best = sorted(comm, key=lambda n: -sub.degree(n, weight="weight"))[:max_size]
        groups.append([int(pool[n]) for n in best])
    return sorted(groups, key=len, reverse=True)


def find_new_clusters(ann, T: dict, mu: float, sd: float, rgb: dict, ask: Ask,
                      z_min: float = 2.0, k: int = 5, max_calls: int = 50,
                      verify: bool = True, max_size: int = 12,
                      progress: Callable[[float, str], None] | None = None) -> dict:
    """Stages 4-5: proposals of new clusters (nothing is changed in `ann`).

      a. look-alike groups (propose_groups) decide which crops are sent
         together; the VLM reads EACH crop on its own (read_each);
      b. tokens are grouped by their reading (normalised), across all
         groups - two occurrences of a word that fell into different
         look-alike groups meet here;
      c. with verify=True every member is checked visually against the
         others (verify_word); only members the VLM confirms are kept, and a
         proposal needs at least two.
    A proposal whose reading matches an existing open cluster's name is
    offered for that cluster (`match`). max_calls limits the reading calls;
    verification adds about one call per proposed word."""
    groups = propose_groups(ann, T, mu, sd, z_min, k)[:max_calls]
    names = {}
    for cid, c in ann.clusters.items():
        if c["status"] in OPEN and label_of(c):
            names.setdefault(norm_text(label_of(c)), cid)
    crops: dict[int, np.ndarray] = {}

    def crop(r: int) -> np.ndarray:
        if r not in crops:
            w = token_word(T, r)
            crops[r] = crop_box(rgb[w["page"]], w["box"])
        return crops[r]

    # a. one reading per crop
    reading: dict[int, str] = {}
    total = len(groups)
    for n, g in enumerate(groups, 1):
        if progress:
            progress(n / max(total, 1) * (0.5 if verify else 1.0), f"reading group {n}/{total}")
        for r, rd in zip(g, read_each([numbered(crop(r), m) for m, r in enumerate(g, 1)], ask)):
            text = rd.get("expanded") or rd.get("diplomatic")
            if text and r not in reading:
                reading[r] = text
    # b. same reading -> same proposal
    buckets: dict[str, list[int]] = {}
    for r, text in reading.items():
        key = norm_text(text)
        if len(key) >= 3:
            buckets.setdefault(key, []).append(r)
    buckets = {key: rows[:max_size] for key, rows in buckets.items() if len(rows) >= 2}
    # c. visual confirmation
    proposals, n_verify = [], 0
    for b, (key, rows) in enumerate(buckets.items(), 1):
        kept = rows
        if verify:
            if progress:
                progress(0.5 + 0.5 * b / len(buckets), f"checking '{key}' ({len(rows)} words)")
            kept = []
            for r in rows:
                others = [o for o in rows if o != r][:4]
                n_verify += 1
                ok, _ = verify_word(crop(r), [crop(o) for o in others], ask)
                if ok:
                    kept.append(r)
                if len(rows) == 2:                       # a pair needs only one check
                    kept = rows if ok else []
                    break
        if len(kept) < 2:
            continue
        shown = Counter(reading[r] for r in kept).most_common(1)[0][0]
        match = names.get(key)
        proposals.append(dict(
            id=f"p{len(proposals) + 1}", reading=shown, match=match,
            match_number=ann.clusters[match]["number"] if match else None,
            members=[T["ids"][r] for r in kept], words=[token_word(T, r) for r in kept],
            dropped=[T["ids"][r] for r in rows if r not in kept], verified=verify))
    return dict(groups=len(groups), calls=len(groups) + n_verify, read=len(reading),
                proposals=sorted(proposals, key=lambda p: -len(p["members"])))


def create_proposal(stem: str, get_ann, p: dict, who: str, into: str | None) -> None:
    """Button callback: the proposal's words become a new cluster (into=None)
    or join cluster `into`. Words that meanwhile belong to a cluster are skipped."""
    import streamlit as st
    ann = get_ann(stem)
    free = [w for w in p["words"] if w["id"] not in ann.data["words"]]
    if len(free) < (1 if into else 2):
        st.session_state["flash"] = ("error", "these words are already in clusters")
        return
    target = into
    for w in free:
        target = ann.add_word(dict(w, origin="vlm"), target, who)
        if p.get("reading"):           # names follow from readings (update_auto_names)
            ann.data["readings"][w["id"]] = dict(diplomatic=p["reading"], expanded=p["reading"],
                                                 confidence=None, cluster_check=True)
    ann.update_auto_names()
    ann.log(who, "vlm_new_cluster" if into is None else "vlm_join", target,
            words=[w["id"] for w in free], reading=p.get("reading"))
    ann.save()
    st.session_state["flash"] = ("ok", f"cluster #{ann.clusters[target]['number']}: "
                                       f"{len(free)} word(s) added")

# =========================================================================== #
# 4. Streamlit panel
# =========================================================================== #
def backend_settings(st) -> Ask | None:
    """Model choice in the recall panel; returns the ask() to use, or None
    while a needed API key is missing."""
    import os
    backend = st.radio("Model", ["Qwen (API)", "Qwen (local GPU)", "DeepSeek"],
                       key="vr_backend", horizontal=True)
    if backend == "DeepSeek":
        if not st.session_state.get("ds_key"):
            st.caption("Add the DeepSeek API key under *DeepSeek settings* "
                       "(model: deepseek-flash).")
            return None
        os.environ["DEEPSEEK_API_KEY"] = st.session_state["ds_key"]
        if tc is not None:
            tc.MODEL = st.session_state.get("ds_model", tc.MODEL)
        return deepseek_ask_json
    if backend == "Qwen (local GPU)":
        model = st.text_input("Hugging Face model", QWEN_LOCAL_MODEL, key="vr_qlm")
        if st.session_state.get("vr_qlocal_id") != model:      # load once per model
            st.session_state["vr_qlocal"] = QwenLocal(model)
            st.session_state["vr_qlocal_id"] = model
        return st.session_state["vr_qlocal"]
    st.session_state.setdefault("vr_qkey", os.environ.get("DASHSCOPE_API_KEY", ""))
    key = st.text_input("DashScope API key", type="password", key="vr_qkey")
    model = st.text_input("Model", QWEN_API_MODEL, key="vr_qm")
    url = st.text_input("Endpoint", QWEN_API_URL, key="vr_qurl",
                        help="Singapore by default; Beijing: "
                             "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not key:
        st.caption("Enter your Alibaba Cloud Model Studio (DashScope) API key.")
        return None
    return openai_compatible_ask(url, model, key)


def recall_panel(stem: str, get_ann, T: dict | None, mu: float, sd: float,
                 words: dict, rgb: dict, ask: Ask | None = None) -> None:
    """Sidebar expander: find words no cluster holds and let the VLM add
    them. Re-reads the annotation before writing (like spotting_app.do).
    With ask=None the model is chosen in the panel (Qwen API / local /
    DeepSeek)."""
    import streamlit as st

    with st.sidebar.expander("VLM recall (missed words)"):
        if T is None:
            st.caption("Needs cands_<tag>.npz from the spotting run.")
            return
        z_min = st.slider("Candidate z (lower = more recall, more calls)", -1.0, 4.0, 1.0,
                          0.25, key="vr_z")
        ratio = st.slider("Width ratio to the cluster", 1.1, 2.5, 1.6, 0.1, key="vr_w",
                          help="looser than the spotting's 1.3: abbreviated forms are shorter")
        max_calls = st.number_input("Max VLM calls", 1, 2000, 100, key="vr_max")
        ann = get_ann(stem)
        n = len(recall_candidates(ann, T, mu, sd, z_min, ratio))
        st.caption(f"{n} (token, cluster) candidates at these settings")
        dry = st.checkbox("Dry run (log only, add nothing)", False, key="vr_dry")
        ask = ask or backend_settings(st)
        if ask is None:
            return
        if st.button("Find missed words", type="primary", disabled=n == 0, key="vr_go"):
            bar = st.progress(0.0)
            res = run_recall(ann, T, mu, sd, words, rgb, ask, z_min, ratio,
                             max_calls=int(max_calls), dry_run=dry,
                             progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
            if not dry and res["added"]:
                ann.log("vlm", "vlm_recall", None, added=res["added"], calls=res["calls"],
                        z_min=z_min, width_ratio=ratio)
                ann.update_auto_names()
                ann.save()
            st.session_state["vr_last"] = res
            st.rerun()
        st.divider()
        st.markdown("**New clusters** (words that recur but are in no cluster)")
        nz_ = st.slider("Similarity z for grouping", 0.5, 5.0, 2.0, 0.25, key="vr_nz",
                        help="lower = more groups proposed, more VLM calls")
        nk = st.slider("Neighbours per word (k)", 2, 15, 5, key="vr_nk")
        ng = st.number_input("Max groups sent to the VLM", 1, 1000, 30, key="vr_ng")
        vchk = st.checkbox("Confirm every proposal visually (second VLM check)", True,
                           key="vr_nv", help="about one extra call per proposed word; "
                                             "drops words the VLM does not confirm")
        if st.button("Find new clusters", key="vr_new_go"):
            bar = st.progress(0.0)
            st.session_state["vr_new"] = find_new_clusters(
                get_ann(stem), T, mu, sd, rgb, ask, nz_, nk, int(ng), verify=vchk,
                progress=lambda f, t: bar.progress(min(f, 1.0), text=t))
            st.rerun()
        new = st.session_state.get("vr_new")
        if new:
            st.success(f"{new['groups']} group(s) read · {new['calls']} VLM call(s) · "
                       f"{len(new['proposals'])} proposal(s) - see the review tab")
        res = st.session_state.get("vr_last")
        if res:
            st.success(f"last run: {res['calls']} calls · {res['added']} word(s) "
                       f"{'would be ' if dry else ''}added")
            st.download_button("Download log (JSON)", json.dumps(res["log"], indent=1),
                               f"vlm_recall_{stem}.json", "application/json", key="vr_dl")


# =========================================================================== #
# 5. Review: what the VLM saw and decided, cluster by cluster
# =========================================================================== #
def log_rows(log: list[dict], words: dict) -> list[dict]:
    """Normalise log rows (also logs of earlier versions, which stored only
    the word id): every row gets `wid` and, where known, `word` (geometry)."""
    rows = []
    for r in log:
        r = dict(r)
        if isinstance(r.get("word"), str):              # old format: word = id
            r["wid"] = r.pop("word")
        r.setdefault("exemplars", [])
        if not isinstance(r.get("word"), dict):
            r["word"] = words.get(r["wid"])
        rows.append(r)
    return rows


def by_cluster(rows: list[dict]) -> list[tuple[int, list[dict]]]:
    """Rows grouped by cluster number, clusters with most accepts first."""
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["number"], []).append(r)
    for g in groups.values():
        g.sort(key=lambda r: (not r["accepted"], -r["z"]))
    return sorted(groups.items(), key=lambda kv: (-sum(r["accepted"] for r in kv[1]), kv[0]))


def word_img(word: dict | None, rgb: dict) -> np.ndarray | None:
    """Crop of a word from the page image its box refers to (None if unknown)."""
    if not word or word.get("img", word.get("page")) not in rgb:
        return None
    return crop_box(rgb[word.get("img", word["page"])], word.get("box"))


def html_report(rows: list[dict], ann, words: dict, rgb: dict, title: str) -> str:
    """Self-contained HTML (images embedded): per cluster, its members as
    shown to the VLM, then every candidate with the verdict."""
    def img(w):
        a = word_img(w, rgb)
        return f'<img src="{data_url(a)}">' if a is not None else "<i>no image</i>"

    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>"
             "<style>body{font-family:sans-serif;margin:24px}h2{margin-top:32px}"
             ".row{display:flex;flex-wrap:wrap;gap:10px}.t{border:1px solid #ccc;padding:6px;"
             "font-size:12px;text-align:center}.t img{height:60px;display:block;margin:auto}"
             ".yes{border:3px solid #2a2}.no{border:3px solid #c33}</style></head><body>"
             f"<h1>{title}</h1>"]
    for number, group in by_cluster(rows):
        cid = group[0]["cluster"]
        c = ann.clusters.get(cid, {})
        n_yes = sum(r["accepted"] for r in group)
        parts.append(f"<h2>Cluster #{number} · {group[0].get('label') or label_of(c) or '–'}"
                     f" · {n_yes}/{len(group)} accepted</h2><p>Members shown to the VLM:</p>"
                     "<div class='row'>")
        ex = list(dict.fromkeys(w for r in group for w in r["exemplars"])) or \
            ann.members(cid)[:6]
        parts += [f"<div class='t'>{img(words.get(w))}{w}</div>" for w in ex]
        parts.append("</div><p>Candidates:</p><div class='row'>")
        for r in group:
            parts.append(f"<div class='t {'yes' if r['accepted'] else 'no'}'>{img(r['word'])}"
                         f"{'✅' if r['accepted'] else '❌'} {r.get('reading') or ''}<br>"
                         f"z {r['z']} · conf {r.get('confidence')}<br>{r['wid']}</div>")
        parts.append("</div>")
    return "".join(parts) + "</body></html>"


def _act(stem: str, get_ann, action: str, *args) -> None:
    """Button callback: re-read the annotation, apply one change, save."""
    import streamlit as st
    ann = get_ann(stem)
    try:
        getattr(ann, action)(*args)
    except (ValueError, KeyError) as e:
        st.session_state["flash"] = ("error", str(e))
        return
    ann.update_auto_names()
    ann.save()
    st.session_state["flash"] = ("ok", "done")


def review_view(stem: str, get_ann, words: dict, rgb: dict) -> None:
    """Main-area review of a recall run (the last one, or a downloaded log):
    per cluster, the members the VLM was shown and every candidate with its
    verdict, reading and z. Candidates can be added (e.g. after a dry run,
    or to overrule a 'no') or taken out again."""
    import streamlit as st

    mode = st.radio("Review", ["Missed words (added to existing clusters)",
                               "New clusters (proposals)"], horizontal=True, key="vr_mode")
    if mode.startswith("New"):
        new_clusters_view(stem, get_ann, rgb)
        return
    up = st.file_uploader("Or open a saved log (vlm_recall_*.json)", type="json",
                          key="vr_log_up")
    if up is not None:
        log = json.loads(up.getvalue())
    else:
        log = (st.session_state.get("vr_last") or {}).get("log")
    if not log:
        st.info("Run *Find missed words* in the sidebar (VLM recall), or open a saved log.")
        return
    ann = get_ann(stem)
    rows = log_rows(log, words)
    show = st.radio("Show", ["all", "accepted", "rejected"], horizontal=True, key="vr_show")
    if show != "all":
        rows = [r for r in rows if r["accepted"] == (show == "accepted")]
    st.caption(f"{len(rows)} candidate(s) in {len({r['number'] for r in rows})} cluster(s)")
    st.download_button("Download as HTML report",
                       html_report(rows, ann, words, rgb, f"VLM recall · {stem}"),
                       f"vlm_recall_{stem}.html", "text/html", key="vr_html")
    who = st.session_state.get("annotator", "").strip()

    for number, group in by_cluster(rows):
        cid = group[0]["cluster"]
        n_yes = sum(r["accepted"] for r in group)
        with st.expander(f"Cluster #{number} · {group[0].get('label') or '–'} · "
                         f"{n_yes}/{len(group)} accepted", expanded=len(group) <= 4):
            st.caption("Members shown to the VLM")
            ex = list(dict.fromkeys(w for r in group for w in r["exemplars"])) or \
                ann.members(cid)[:6]
            cols = st.columns(max(len(ex), 1))
            for col, w in zip(cols, ex):
                a = word_img(words.get(w), rgb)
                if a is not None:
                    col.image(a, caption=w)
            st.caption("Candidates")
            cols = st.columns(4)
            for k, r in enumerate(group):
                with cols[k % 4].container(border=True):
                    a = word_img(r["word"], rgb)
                    if a is not None:
                        st.image(a)
                    else:
                        st.caption("(no image in this log: re-run to record boxes)")
                    st.markdown(f"{'✅' if r['accepted'] else '❌'} **{r.get('reading') or '?'}**"
                                f"  \n:gray[z {r['z']} · conf {r.get('confidence')}]")
                    st.caption(r["wid"])
                    state = ann.data["words"].get(r["wid"])
                    inside = state is not None and not state["removed"] \
                        and state["cluster"] == cid
                    if not who:
                        continue
                    if inside:
                        st.button("Take out", key=f"vr_out_{r['wid']}_{cid}",
                                  on_click=_act, args=(stem, get_ann, "remove_word", r["wid"], who))
                    elif state is None and r["word"]:
                        st.button(f"Add to #{number}", key=f"vr_add_{r['wid']}_{cid}",
                                  type="primary" if r["accepted"] else "secondary",
                                  on_click=_act,
                                  args=(stem, get_ann, "add_word",
                                        dict(r["word"], origin="vlm"), cid, who))


def new_clusters_view(stem: str, get_ann, rgb: dict) -> None:
    """Proposed new clusters (from find_new_clusters): the words of each
    proposal with the VLM's reading; the annotator creates the cluster, adds
    the words to the matching existing cluster, or leaves the proposal."""
    import streamlit as st

    up = st.file_uploader("Or open saved proposals (vlm_new_clusters_*.json)", type="json",
                          key="vr_new_up")
    res = json.loads(up.getvalue()) if up is not None else st.session_state.get("vr_new")
    if not res or not res.get("proposals"):
        st.info("Run *Find new clusters* in the sidebar (VLM recall), or open saved proposals.")
        return
    ann = get_ann(stem)
    who = st.session_state.get("annotator", "").strip()
    props = res["proposals"]
    done = [all(m in ann.data["words"] for m in p["members"]) for p in props]
    hide = st.toggle("Hide proposals already accepted", True, key="vr_new_hide")
    st.caption(f"{len(props)} proposal(s) · {sum(done)} accepted · "
               f"{sum(p['match'] is not None for p in props)} match an existing cluster")
    st.download_button("Download proposals (JSON)", json.dumps(res, indent=1),
                       f"vlm_new_clusters_{stem}.json", "application/json", key="vr_new_dl")
    if not who:
        st.warning("Enter your name as annotator in the sidebar to accept proposals.")
    for p, d in zip(props, done):
        if d and hide:
            continue
        head = f"**{p['id']} · {p.get('reading') or '?'}** · {len(p['words'])} words"
        if p["match"] is not None:
            head += f" · same reading as existing cluster **#{p['match_number']}**"
        with st.container(border=True):
            st.markdown(head + (" · ✅ accepted" if d else ""))
            cols = st.columns(min(len(p["words"]), 6))
            for k, w in enumerate(p["words"]):
                a = word_img(w, rgb)
                if a is not None:
                    cols[k % len(cols)].image(a, caption=w["id"])
            if d or not who:
                continue
            b1, b2 = st.columns(2)
            if p["match"] is not None and p["match"] in ann.clusters:
                b1.button(f"Add to #{p['match_number']}", key=f"vr_pj_{p['id']}",
                          type="primary", on_click=create_proposal,
                          args=(stem, get_ann, p, who, p["match"]))
                b2.button("New cluster instead", key=f"vr_pn_{p['id']}",
                          on_click=create_proposal, args=(stem, get_ann, p, who, None))
            else:
                b1.button("Create cluster", key=f"vr_pn_{p['id']}", type="primary",
                          on_click=create_proposal, args=(stem, get_ann, p, who, None))