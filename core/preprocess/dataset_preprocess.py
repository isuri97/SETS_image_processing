"""
Keep only the expected columns in annotator or symbol CSVs.

Set the paths in the CONFIG section below, then press Run in PyCharm.

The schema (annotator or symbol) is detected from the file's headers, and
any column not in that schema is dropped. Columns are written in the
schema's order. Expected columns that are missing are reported, not added.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =========================================================================== #
# CONFIG — edit these, then press Run
# =========================================================================== #
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILES = [
    SCRIPT_DIR / "labels_result.csv",
    SCRIPT_DIR / "symbols.csv",
]

OVERWRITE = False   # True: replace the original files; False: write <name>_clean.csv
FORCE_SCHEMA = None  # None to auto-detect, or "annotator" / "symbol"
# =========================================================================== #

SCHEMAS = {
    "annotator": ["concept", "name", "page", "x0", "y0", "x1", "y1"],
    "symbol": ["source_file", "page", "symbol", "kind", "relation",
               "ox0", "oy0", "ox1", "oy1"],
}

def detect_schema(columns: list[str]) -> str:
    """Pick the schema that shares the most columns with the file."""
    cols = set(columns)
    return max(SCHEMAS, key=lambda name: len(cols & set(SCHEMAS[name])))


def clean(df: pd.DataFrame, schema: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return the trimmed frame, the dropped columns and the missing ones."""
    expected = SCHEMAS[schema]
    keep = [c for c in expected if c in df.columns]
    dropped = [c for c in df.columns if c not in expected]
    missing = [c for c in expected if c not in df.columns]
    return df[keep], dropped, missing


def process(path: Path) -> None:
    if not path.exists():
        print(f"[skip] not found: {path}")
        return

    # dtype=str preserves values exactly (IDs, JSON fields, timestamps)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    schema = FORCE_SCHEMA or detect_schema(list(df.columns))
    out_df, dropped, missing = clean(df, schema)

    target = path if OVERWRITE else path.with_name(f"{path.stem}_clean.csv")
    out_df.to_csv(target, index=False)

    print(f"{path.name} [{schema}] -> {target.name}")
    print(f"  dropped: {dropped or 'none'}")
    if missing:
        print(f"  missing expected columns: {missing}")


if __name__ == "__main__":
    for file in INPUT_FILES:
        process(Path(file))