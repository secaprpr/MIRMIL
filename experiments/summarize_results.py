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
    parser.add_argument("--aggregate-output")
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
    metric_columns = [
        "test_macro_auc",
        "test_acc",
        "test_bacc",
        "test_macro_f1",
        "selected_ratio_mean",
        "full_macro_auc",
        "complement_macro_auc",
        "necessity_confidence_drop",
    ]
    available_metrics = [
        column for column in metric_columns if column in result.columns
    ]
    aggregate = result.groupby("model")[available_metrics].agg(["mean", "std", "count"])
    aggregate.columns = [
        "_".join(column).rstrip("_") for column in aggregate.columns.to_flat_index()
    ]
    aggregate = aggregate.reset_index()
    if args.output:
        result.to_csv(args.output, index=False)
    if args.aggregate_output:
        aggregate.to_csv(args.aggregate_output, index=False)
    print("Per-run results")
    print(result.to_string(index=False))
    print("\nAggregate results")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
