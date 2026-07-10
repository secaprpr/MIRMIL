"""Build BRACS 3-class split CSVs from existing 7-class feature split CSVs.

Mapping follows the BRACS group labels encoded in the source manifest:

- BT / benign: N, PB, UDH -> 0
- AT / atypical: FEA, ADH -> 1
- MT / malignant: DCIS, IC -> 2

This script only rewrites slide-level labels in CSV manifests. It does not touch
feature files, feature extraction code, or raw WSI data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEVEN_TO_THREE = {
    0: 0,  # N -> BT
    1: 0,  # PB -> BT
    2: 0,  # UDH -> BT
    3: 1,  # FEA -> AT
    4: 1,  # ADH -> AT
    5: 2,  # DCIS -> MT
    6: 2,  # IC -> MT
}


def convert_split(input_csv: Path, output_csv: Path) -> dict[str, dict[int, int]]:
    df = pd.read_csv(input_csv)
    counts: dict[str, dict[int, int]] = {}
    for group in ("train", "val", "test"):
        label_col = f"{group}_label"
        path_col = f"{group}_slide_path"
        if label_col not in df.columns or path_col not in df.columns:
            continue
        mask = df[path_col].notna()
        labels = df.loc[mask, label_col].astype(int)
        mapped = labels.map(SEVEN_TO_THREE)
        if mapped.isna().any():
            bad = sorted(labels[mapped.isna()].unique().tolist())
            raise ValueError(f"Unknown labels in {input_csv}: {bad}")
        df.loc[mask, label_col] = mapped.astype(int).values
        counts[group] = (
            df.loc[mask, label_col].astype(int).value_counts().sort_index().to_dict()
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(
            "/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata"
        ),
    )
    args = parser.parse_args()

    jobs = [
        (
            "BRACS_r50_split_official_train_val.csv",
            "BRACS3_r50_split_official_train_val.csv",
        ),
        (
            "BRACS_r50_split_official_train_val_h5.csv",
            "BRACS3_r50_split_official_train_val_h5.csv",
        ),
        (
            "BRACS_r50_split_official_full.csv",
            "BRACS3_r50_split_official_full.csv",
        ),
        (
            "BRACS_r50_split_official_full_h5.csv",
            "BRACS3_r50_split_official_full_h5.csv",
        ),
        (
            "BRACS_uni_split_official_train_val.csv",
            "BRACS3_uni_split_official_train_val.csv",
        ),
        (
            "BRACS_uni_split_official_train_val_h5.csv",
            "BRACS3_uni_split_official_train_val_h5.csv",
        ),
        (
            "BRACS_uni_split_official_full.csv",
            "BRACS3_uni_split_official_full.csv",
        ),
        (
            "BRACS_uni_split_official_full_h5.csv",
            "BRACS3_uni_split_official_full_h5.csv",
        ),
    ]
    for src_name, dst_name in jobs:
        src = args.metadata_dir / src_name
        dst = args.metadata_dir / dst_name
        counts = convert_split(src, dst)
        print(f"{src.name} -> {dst.name}: {counts}")


if __name__ == "__main__":
    main()
