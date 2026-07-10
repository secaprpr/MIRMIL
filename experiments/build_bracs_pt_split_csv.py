"""Build BRACS official split CSVs that point to existing PT feature files.

The official split and labels are copied unchanged from the H5 split CSVs.
Only feature paths are rewritten from ``<feature>/h5_files/*.h5`` to
``<feature>/pt_files/*.pt`` for faster MIL training I/O.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PATH_COLUMNS = ("train_slide_path", "val_slide_path", "test_slide_path")


def rewrite_path(value):
    if pd.isna(value):
        return value
    path = Path(str(value))
    return str(path.parent.parent / "pt_files" / f"{path.stem}.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for column in PATH_COLUMNS:
        df[column] = df[column].map(rewrite_path)

    missing = []
    for column in PATH_COLUMNS:
        for value in df[column].dropna():
            if not Path(value).is_file():
                missing.append(value)
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} rewritten PT feature files are missing. "
            f"First missing paths:\n{preview}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
