#!/usr/bin/env python3
from __future__ import print_function

from collections import OrderedDict
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search_investigation.data import load_breast_clinical, modality_dimensions
from search_investigation.paths import MANIFESTS_DIR, ensure_project_dirs
from search_investigation.tasks import list_task_names, task_label_counts


def main():
    ensure_project_dirs()
    breast = load_breast_clinical()

    patient_manifest = breast.copy()
    patient_manifest["RZ_ID"] = patient_manifest["RZ_ID"].astype(str)
    patient_manifest.to_csv(str(MANIFESTS_DIR / "breast_patient_manifest.csv"), index=False)

    modality_manifest = pd.DataFrame(modality_dimensions())
    modality_manifest.to_csv(str(MANIFESTS_DIR / "modality_manifest.csv"), index=False)

    task_rows = []
    for task_name in list_task_names():
        counts, labeled_count = task_label_counts(patient_manifest, task_name)
        task_rows.append(
            OrderedDict(
                [
                    ("task_name", task_name),
                    ("labeled_patients", labeled_count),
                    ("class_balance", dict(counts)),
                ]
            )
        )
    task_manifest = pd.DataFrame(task_rows)
    task_manifest.to_csv(str(MANIFESTS_DIR / "task_manifest.csv"), index=False)

    print("Wrote:", MANIFESTS_DIR / "breast_patient_manifest.csv")
    print("Wrote:", MANIFESTS_DIR / "modality_manifest.csv")
    print("Wrote:", MANIFESTS_DIR / "task_manifest.csv")


if __name__ == "__main__":
    main()
