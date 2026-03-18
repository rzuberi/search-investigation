#!/usr/bin/env python3
from __future__ import print_function

import argparse
import csv
import subprocess
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_investigation.paths import ARRAY_DERIVED_DIR, CONFIG_DIR, SLURM_DIR, SMOKE_TESTS_DIR, ensure_dir, ensure_project_dirs


RESOURCE_PRESETS = {
    "light": {"cpus": 8, "mem": "32G", "time": "02:00:00"},
    "medium": {"cpus": 16, "mem": "64G", "time": "04:00:00"},
    "heavy": {"cpus": 32, "mem": "128G", "time": "08:00:00"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["tier1", "tier2", "tier3"], default="tier1")
    parser.add_argument("--partition", default="epyc")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def load_matrix_config():
    with open(str(CONFIG_DIR / "smoke_matrix.yaml")) as handle:
        return yaml.safe_load(handle)


def render_job_script(job_name, partition, preset, command, output_dir):
    return "\n".join(
        [
            "#!/bin/bash",
            "#SBATCH -J {0}".format(job_name),
            "#SBATCH -p {0}".format(partition),
            "#SBATCH --cpus-per-task={0}".format(preset["cpus"]),
            "#SBATCH --mem={0}".format(preset["mem"]),
            "#SBATCH --time={0}".format(preset["time"]),
            "#SBATCH -o {0}".format(output_dir / (job_name + ".out")),
            "#SBATCH -e {0}".format(output_dir / (job_name + ".err")),
            "",
            "set -euo pipefail",
            "cd /mnt/scratchc/fmlab/zuberi01/phd/search-investigation",
            command,
            "",
        ]
    )


def main():
    args = parse_args()
    ensure_project_dirs()
    matrix = load_matrix_config()

    tasks = list(matrix["tiers"][args.tier])
    jobs = []

    for modality in matrix["modalities"]["core"]:
        models = matrix["model_sets"]["low_dim"] if modality != "bridges_genotypes" else matrix["model_sets"]["high_dim"]
        for task in tasks:
            for model in models:
                jobs.append((task, modality, model))

    if args.tier in ("tier1", "tier2"):
        for modality in matrix["modalities"]["arrays"]:
            if not (ARRAY_DERIVED_DIR / "{0}_matrix.tsv.gz".format(modality)).exists():
                continue
            models = matrix["model_sets"]["high_dim"]
            for task in tasks:
                if task not in matrix["array_tasks"]:
                    continue
                for model in models:
                    jobs.append((task, modality, model))

    ensure_dir(SLURM_DIR)
    manifest_path = SLURM_DIR / "run_manifest_{0}.csv".format(args.tier)
    with open(str(manifest_path), "w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "modality", "model", "seed", "script_path", "submitted", "job_id"])
        for task, modality, model in jobs:
            job_name = "{0}-{1}-{2}".format(task[:18], modality[:18], model[:18]).replace("_", "-")
            run_dir = SMOKE_TESTS_DIR / task / modality / model / ("seed_{0}".format(args.seed))
            ensure_dir(run_dir)
            command = (
                "python3 scripts/run_smoke_test.py --task {task} --modality {modality} "
                "--model {model} --seed {seed} --n-jobs {n_jobs} --outdir {outdir}"
            ).format(
                task=task,
                modality=modality,
                model=model,
                seed=args.seed,
                n_jobs=args.n_jobs,
                outdir=run_dir,
            )
            preset = RESOURCE_PRESETS[matrix["resource_classes"][modality]]
            script_path = SLURM_DIR / "{0}.sbatch".format(job_name)
            with open(str(script_path), "w") as script_handle:
                script_handle.write(render_job_script(job_name, args.partition, preset, command, SLURM_DIR))
            submitted = "no"
            job_id = ""
            if args.submit:
                output = subprocess.check_output(["sbatch", str(script_path)]).decode("utf-8").strip()
                submitted = "yes"
                job_id = output.split()[-1]
            writer.writerow([task, modality, model, args.seed, str(script_path), submitted, job_id])

    print("Wrote:", manifest_path)
    print("Jobs:", len(jobs))


if __name__ == "__main__":
    main()
