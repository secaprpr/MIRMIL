#!/usr/bin/env python
"""Prepare RRT-MIL Survival CSVs from MIRMIL prognosis split files.

RRT-MIL Survival expects a patient-level CSV with columns:
Study, ID, Event, Status, WSI

The MIRMIL prognosis split files are wide train/val/test tables and may contain
multiple slides per patient. This script converts them to patient-level records
and joins multiple WSI feature paths with semicolons, matching RRT-MIL's
TCGA_Survival dataset implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SPLITS = ("train", "val", "test")


def _pick_path_column(df: pd.DataFrame, split: str) -> str:
    candidates = [
        f"{split}_feature_path",
        f"{split}_slide_path",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Cannot find feature path column for split={split}; tried {candidates}")


def wide_to_long(path: Path, study: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        path_col = _pick_path_column(df, split)
        required = [path_col, f"{split}_patient_id", f"{split}_time_months", f"{split}_event"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"{path} missing columns for {split}: {missing}")
        sub = df[required].copy()
        sub = sub.dropna(subset=[path_col, f"{split}_patient_id", f"{split}_time_months", f"{split}_event"])
        sub = sub.rename(
            columns={
                path_col: "WSI",
                f"{split}_patient_id": "ID",
                f"{split}_time_months": "Event",
                f"{split}_event": "Status",
            }
        )
        sub["Split"] = split
        parts.append(sub)
    long = pd.concat(parts, ignore_index=True)
    long.insert(0, "Study", study)
    long["Event"] = pd.to_numeric(long["Event"], errors="raise")
    long["Status"] = pd.to_numeric(long["Status"], errors="raise").astype(int)
    return long


def aggregate_patient(long: pd.DataFrame) -> pd.DataFrame:
    split_counts = long.groupby("ID")["Split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        examples = ", ".join(leaked.index[:10].tolist())
        raise ValueError(f"Patients appear in multiple splits: {examples}")

    grouped_rows = []
    for patient_id, group in long.groupby("ID", sort=True):
        event_values = group["Event"].drop_duplicates()
        status_values = group["Status"].drop_duplicates()
        if len(event_values) != 1 or len(status_values) != 1:
            raise ValueError(f"Inconsistent prognosis labels for patient {patient_id}")
        paths = []
        for p in group["WSI"].astype(str):
            if p not in paths:
                paths.append(p)
        grouped_rows.append(
            {
                "Study": group["Study"].iloc[0],
                "ID": patient_id,
                "Event": float(event_values.iloc[0]),
                "Status": int(status_values.iloc[0]),
                "WSI": ";".join(paths),
                "Split": group["Split"].iloc[0],
                "NumSlides": len(paths),
            }
        )
    return pd.DataFrame(grouped_rows)


def validate_paths(df: pd.DataFrame) -> tuple[int, list[str]]:
    checked = 0
    missing: list[str] = []
    for joined in df["WSI"].astype(str):
        for p in joined.split(";"):
            checked += 1
            if not Path(p).exists():
                missing.append(p)
    return checked, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    long = wide_to_long(args.input, args.study)
    patient = aggregate_patient(long)
    checked, missing = validate_paths(patient)
    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)}/{checked} feature paths missing. First missing:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    patient.to_csv(args.output, index=False)

    split_summary = patient.groupby("Split")["ID"].count().to_dict()
    event_summary = patient.groupby("Split")["Status"].sum().to_dict()
    print(f"[ok] wrote {args.output}")
    print(f"[summary] patients={len(patient)} slides={checked} splits={split_summary} events={event_summary}")
    if missing:
        print(f"[warning] missing_paths={len(missing)}")


if __name__ == "__main__":
    main()
