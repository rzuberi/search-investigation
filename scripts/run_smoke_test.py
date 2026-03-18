#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_investigation.data import load_breast_clinical, prepare_breast_dataset
from search_investigation.modeling import run_smoke_experiment
from search_investigation.paths import SMOKE_TESTS_DIR, ensure_dir, ensure_project_dirs
from search_investigation.tasks import load_tasks_config, transform_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--outdir")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_project_dirs()
    tasks_config = load_tasks_config()
    task_config = tasks_config[args.task]

    breast = load_breast_clinical()
    x, y = prepare_breast_dataset(
        args.modality,
        args.task,
        breast,
        lambda row: transform_label(args.task, row),
    )
    result, top_features = run_smoke_experiment(
        x=x,
        y=y,
        task_name=args.task,
        task_kind=task_config["kind"],
        model_name=args.model,
        seed=args.seed,
        n_jobs=args.n_jobs,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
    )

    outdir = Path(args.outdir) if args.outdir else SMOKE_TESTS_DIR / args.task / args.modality / args.model / ("seed_{0}".format(args.seed))
    ensure_dir(outdir)
    with open(str(outdir / "metrics.json"), "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    top_features.to_csv(str(outdir / "top_features.csv"), index=False)

    print("Saved:", outdir / "metrics.json")
    print("Saved:", outdir / "top_features.csv")


if __name__ == "__main__":
    main()
