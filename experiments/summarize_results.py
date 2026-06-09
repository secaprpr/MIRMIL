import argparse
import glob
import json
import os

import pandas as pd


def find_best_row(log_path):
    frame = pd.read_csv(log_path)
    test_rows = frame[frame["test_macro_auc"].notna()]
    if len(test_rows):
        return test_rows.iloc[0]
    return frame.loc[frame["val_macro_auc"].idxmax()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        for log_path in glob.glob(
            os.path.join(root, "**", "Log_seed*.csv"), recursive=True
        ):
            row = find_best_row(log_path)
            run_dir = os.path.dirname(log_path)
            diagnostics_path = os.path.join(run_dir, "OT_MIL_diagnostics.json")
            diagnostics = {}
            if os.path.isfile(diagnostics_path):
                with open(diagnostics_path, encoding="utf-8") as file:
                    diagnostics = json.load(file)
            rows.append(
                {
                    "run_dir": run_dir,
                    "model": os.path.basename(os.path.dirname(run_dir)),
                    "epoch": int(row["epoch"]),
                    "val_macro_auc": row["val_macro_auc"],
                    "test_macro_auc": row["test_macro_auc"],
                    "test_acc": row["test_acc"],
                    "test_bacc": row["test_bacc"],
                    "test_macro_f1": row["test_macro_f1"],
                    **diagnostics,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["test_macro_auc", "val_macro_auc"], ascending=False
    )
    if args.output:
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
