"""Process every image in a folder and record the time taken for each one.

Usage
-----
    python batch.py /path/to/images
    python batch.py /path/to/images --skip-done     # only new or previously failed images

Results are written into the same folder:

    <folder>/results/<image name>/   overlay, marks-only, text-only, brace crops, result.json
    <folder>/results/timings.csv     one row per image: status, seconds, mark counts
    <folder>/results/summary.txt     totals and average time per image

Only images directly inside the folder are processed (not subfolders), so the
result images are never picked up on a second run. An image that fails is
recorded with its error message and the batch carries on.
"""
from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

from run import process

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MARK_KINDS = ("rule", "slash", "dash", "circle", "filler")
COLUMNS = ["image", "status", "seconds", "braces", "member_lines", "notes", *MARK_KINDS, "error"]


def find_images(folder: Path) -> list[Path]:
    """Image files directly inside `folder`, in name order."""
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)


def run_one(image: Path, out_root: Path) -> dict:
    """Process one image; return a timings row, whether it succeeded or failed."""
    row = {"image": image.name}
    start = time.perf_counter()
    try:
        counts = process(str(image), out_root)["counts"]
        row.update(status="ok", braces=counts["braces"], member_lines=sum(counts["members"]),
                   notes=counts["notes"], **{k: counts.get(k, 0) for k in MARK_KINDS})
    except Exception as exc:  # one bad image must not stop the batch
        if isinstance(exc, FileNotFoundError) and image.exists():
            message = "file could not be read as an image (corrupt or unsupported format)"
        else:
            message = f"{type(exc).__name__}: {exc}"
        row.update(status="failed", error=message)
        (out_root / f"{image.stem}_error.txt").write_text(traceback.format_exc())
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row


def write_summary(rows: list[dict], path: Path, total: float) -> str:
    """Write and return a short plain-text summary of the batch."""
    ok = [r for r in rows if r["status"] == "ok"]
    times = [r["seconds"] for r in ok]
    lines = [
        f"images processed : {len(rows)}",
        f"succeeded        : {len(ok)}",
        f"failed           : {len(rows) - len(ok)}",
        f"time of this run : {total:.1f} s",
    ]
    if times:
        lines += [
            f"average per image: {sum(times) / len(times):.1f} s",
            f"fastest / slowest: {min(times):.1f} s / {max(times):.1f} s",
        ]
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, help="folder containing the page images")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip images that already have a result.json (failed images are retried)")
    args = ap.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    images = find_images(folder)
    if not images:
        raise SystemExit(f"No images ({', '.join(sorted(IMAGE_TYPES))}) found in {folder}")

    out_root = folder / "results"
    out_root.mkdir(exist_ok=True)
    if args.skip_done:
        images = [p for p in images if not (out_root / p.stem / "result.json").exists()]

    csv_path = out_root / "timings.csv"
    # with --skip-done, keep the rows of earlier runs and add to them
    earlier = []
    if args.skip_done and csv_path.exists():
        with csv_path.open(newline="") as fh:
            earlier = list(csv.DictReader(fh))
        for r in earlier:
            r["seconds"] = float(r["seconds"])
    if not images:
        print("Nothing new to process.")
        return

    rows, t0 = [], time.perf_counter()
    with csv_path.open("a" if earlier else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if not earlier:
            writer.writeheader()
        for i, image in enumerate(images, 1):
            row = run_one(image, out_root)
            writer.writerow(row)
            fh.flush()  # the CSV stays usable if the batch is interrupted
            rows.append(row)
            print(f"[{i}/{len(images)}] {image.name}: {row['status']} in {row['seconds']} s")

    # one row per image: a retried image replaces its earlier row
    done = {r["image"] for r in rows}
    merged = sorted([r for r in earlier if r["image"] not in done] + rows, key=lambda r: r["image"])
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    print(write_summary(merged, out_root / "summary.txt", time.perf_counter() - t0))
    print(f"Results in {out_root}")


if __name__ == "__main__":
    main()