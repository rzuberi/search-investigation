from __future__ import print_function

from collections import Counter

import pandas as pd
import yaml

from .paths import CONFIG_DIR


def load_tasks_config():
    with open(str(CONFIG_DIR / "tasks.yaml")) as handle:
        return yaml.safe_load(handle)["tasks"]


def transform_label(task_name, row):
    if task_name == "grade_high_vs_low":
        value = row.get("Grade")
        if value == 3:
            return "high_3"
        if value in (1, 2):
            return "low_1_2"
        return None
    if task_name == "grade_3class":
        value = row.get("Grade")
        if value in (1, 2, 3):
            return str(int(value))
        return None
    if task_name == "er_status":
        value = row.get("ER_status")
        return value if value in ("P", "N") else None
    if task_name == "pr_status":
        value = row.get("PR_status")
        return value if value in ("P", "N") else None
    if task_name == "her2_status":
        value = row.get("HER2_status")
        return value if value in ("P", "N") else None
    if task_name == "stage_early_vs_late":
        value = row.get("Stage")
        if value in (1, 2):
            return "early_1_2"
        if value in (3, 4):
            return "late_3_4"
        return None
    if task_name == "stage_4class":
        value = row.get("Stage")
        if value in (1, 2, 3, 4):
            return str(int(value))
        return None
    if task_name == "vital_status":
        value = row.get("VitalStatus")
        if value == 0:
            return "alive"
        if value == 1:
            return "dead"
        return None
    if task_name == "cause_specific_outcome":
        vital_status = row.get("VitalStatus")
        cause = row.get("CauseOfDeath")
        if vital_status == 0:
            return "alive"
        if cause == 1:
            return "cancer_death"
        if cause == 3:
            return "other_death"
        return None
    if task_name == "chemo_received":
        value = row.get("Chemotherapy")
        return str(int(value)) if value in (0, 1) else None
    if task_name == "radio_received":
        value = row.get("Radiotherapy")
        return str(int(value)) if value in (0, 1) else None
    if task_name == "surgery_received":
        value = row.get("Surgery")
        return str(int(value)) if value in (0, 1) else None
    if task_name == "screen_detected":
        value = row.get("ScreenDetected")
        return str(int(value)) if value in (0, 1) else None
    raise KeyError("Unknown task: {0}".format(task_name))


def attach_task_labels(frame, task_name):
    labels = frame.apply(lambda row: transform_label(task_name, row), axis=1)
    output = frame.copy()
    output["task_label"] = labels
    return output


def task_label_counts(frame, task_name):
    labeled = attach_task_labels(frame, task_name)
    labeled = labeled[labeled["task_label"].notnull()]
    counts = Counter(labeled["task_label"].tolist())
    return counts, labeled.shape[0]


def list_task_names():
    return list(load_tasks_config().keys())
