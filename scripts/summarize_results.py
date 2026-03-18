#!/usr/bin/env python3
from __future__ import print_function

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_investigation.paths import MANIFESTS_DIR, SMOKE_TESTS_DIR, ensure_project_dirs


def main():
    ensure_project_dirs()
    rows = []
    for metrics_path in SMOKE_TESTS_DIR.glob("**/metrics.json"):
        with open(str(metrics_path)) as handle:
            payload = json.load(handle)
        row = {
            "task_name": payload["task_name"],
            "task_kind": payload["task_kind"],
            "model_name": payload["model_name"],
            "seed": payload["seed"],
            "n_samples": payload["n_samples"],
            "n_features": payload["n_features"],
            "metrics_path": str(metrics_path),
        }
        for key, value in payload["test_metrics"].items():
            if key == "confusion_matrix":
                continue
            row["test_{0}".format(key)] = value
        for key, value in payload["cv_metrics"].items():
            row["cv_{0}_mean".format(key)] = value["mean"]
            row["cv_{0}_std".format(key)] = value["std"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    outpath = MANIFESTS_DIR / "smoke_test_summary.csv"
    summary.to_csv(str(outpath), index=False)
    print("Wrote:", outpath)


if __name__ == "__main__":
    main()
