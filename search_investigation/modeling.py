from __future__ import print_function

import json
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize

from .paths import CONFIG_DIR


def load_model_configs():
    with open(str(CONFIG_DIR / "models.yaml")) as handle:
        return yaml.safe_load(handle)["models"]


def build_estimator(model_name, task_kind, seed, n_jobs):
    model_configs = load_model_configs()
    config = model_configs[model_name]
    family = config["family"]
    params = dict(config["params"])
    if family == "logistic_regression":
        if params.get("solver") == "liblinear" and task_kind != "binary":
            params["solver"] = "lbfgs"
            params["penalty"] = "l2"
        params["random_state"] = seed
        if task_kind != "binary":
            params["multi_class"] = "auto"
        return LogisticRegression(**params)
    if family == "random_forest":
        params["random_state"] = seed
        params["n_jobs"] = n_jobs
        return RandomForestClassifier(**params)
    if family == "lightgbm":
        params["random_state"] = seed
        params["n_jobs"] = n_jobs
        if task_kind == "multiclass":
            params["objective"] = "multiclass"
        else:
            params["objective"] = "binary"
        return lgb.LGBMClassifier(**params)
    raise KeyError("Unsupported model family: {0}".format(family))


def build_pipeline(model_name, task_kind, input_width):
    model_configs = load_model_configs()
    config = model_configs[model_name]
    estimator = build_estimator(model_name, task_kind, seed=0, n_jobs=1)
    select_k = config.get("select_k", {}).get("default")
    if input_width <= 1000:
        select_k = config.get("select_k", {}).get("low_dim", select_k)
    steps = [("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
    if config.get("scale"):
        steps.append(("scaler", StandardScaler()))
    if select_k and input_width > select_k:
        steps.append(("select", SelectKBest(score_func=f_classif, k=min(select_k, input_width))))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _reset_estimator_seed(pipeline, model_name, task_kind, seed, n_jobs):
    pipeline = clone(pipeline)
    pipeline.steps[-1] = ("model", build_estimator(model_name, task_kind, seed=seed, n_jobs=n_jobs))
    return pipeline


def _feature_names_after_pipeline(fitted_pipeline, feature_names):
    names = np.array(feature_names)
    if "variance" in fitted_pipeline.named_steps:
        mask = fitted_pipeline.named_steps["variance"].get_support()
        names = names[mask]
    if "select" in fitted_pipeline.named_steps:
        mask = fitted_pipeline.named_steps["select"].get_support()
        names = names[mask]
    return names


def _extract_top_features(fitted_pipeline, feature_names, max_features=25):
    names = _feature_names_after_pipeline(fitted_pipeline, feature_names)
    model = fitted_pipeline.named_steps["model"]
    rows = []
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        scores = np.max(np.abs(coefficients), axis=0)
        order = np.argsort(scores)[::-1][:max_features]
        for index in order:
            rows.append({"feature": names[index], "score": float(scores[index])})
    elif hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_)
        order = np.argsort(scores)[::-1][:max_features]
        for index in order:
            rows.append({"feature": names[index], "score": float(scores[index])})
    return pd.DataFrame(rows)


def _scoring(task_kind):
    scoring = {"balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro"}
    if task_kind == "binary":
        scoring["roc_auc"] = "roc_auc"
        scoring["average_precision"] = "average_precision"
    else:
        scoring["roc_auc_ovr"] = "roc_auc_ovr"
    return scoring


def _safe_cv_splits(y, requested_splits):
    counts = Counter(y)
    smallest_class = min(counts.values())
    return max(2, min(requested_splits, smallest_class))


def _metrics_from_predictions(task_kind, y_true, y_pred, y_proba, labels):
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if task_kind == "binary":
        positive_index = 1 if y_proba.shape[1] > 1 else 0
        positive_scores = y_proba[:, positive_index]
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        metrics["average_precision"] = float(average_precision_score(y_true, positive_scores))
    else:
        y_true_binary = label_binarize(y_true, classes=labels)
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_true_binary, y_proba, multi_class="ovr", average="macro")
        )
    return metrics


def run_smoke_experiment(x, y, task_name, task_kind, model_name, seed, n_jobs, test_size, cv_splits, cv_repeats):
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)
    labels = list(range(len(label_encoder.classes_)))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        encoded_y,
        test_size=test_size,
        stratify=encoded_y,
        random_state=seed,
    )

    pipeline = build_pipeline(model_name, task_kind, x.shape[1])
    pipeline = _reset_estimator_seed(pipeline, model_name, task_kind, seed, n_jobs)

    splits = _safe_cv_splits(y_train, cv_splits)
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=cv_repeats, random_state=seed)
    cv_results = cross_validate(
        clone(pipeline),
        x_train,
        y_train,
        cv=cv,
        scoring=_scoring(task_kind),
        n_jobs=1,
        return_train_score=False,
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)
    metrics = _metrics_from_predictions(task_kind, y_test, y_pred, y_proba, labels)

    cv_summary = {}
    for key, values in cv_results.items():
        if key.startswith("test_"):
            name = key.replace("test_", "")
            cv_summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    top_features = _extract_top_features(pipeline, x.columns.tolist())
    result = {
        "task_name": task_name,
        "task_kind": task_kind,
        "model_name": model_name,
        "seed": seed,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "train_samples": int(x_train.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "label_counts": {label_encoder.classes_[i]: int(np.sum(encoded_y == i)) for i in labels},
        "test_metrics": metrics,
        "cv_metrics": cv_summary,
        "classes": list(label_encoder.classes_),
    }
    return result, top_features
