from __future__ import annotations

import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy.exc import SQLAlchemyError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from ingest_dataset import load_dataset
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import ModelVersion
from app.ml.features import CATEGORICAL_FEATURES, EXCLUDED_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES, TARGETS


def make_pipeline(kind: str, model):
    preprocessing = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def regression_metrics(actual, predicted) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    data, ingestion_check = load_dataset()
    features = data[MODEL_FEATURES].copy()
    features["supplier_tier"] = features["supplier_tier"].astype(str)
    features["has_backup_supplier"] = features["has_backup_supplier"].astype(str)
    train_and_valid_x, test_x = train_test_split(np.arange(len(data)), test_size=0.15, random_state=42)
    train_x, valid_x = train_test_split(train_and_valid_x, test_size=0.1764705882, random_state=42)  # 70/15/15 overall
    output_dir = settings.path(settings.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.path(settings.model_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"dataset_rows": int(len(data)), "ingestion_validation": ingestion_check, "train_rows": int(len(train_x)), "validation_rows": int(len(valid_x)), "test_rows": int(len(test_x)), "features_used": MODEL_FEATURES, "features_excluded_for_leakage": EXCLUDED_FEATURES, "models": {}}
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report["trained_at"] = version

    for key, (target_column, kind) in TARGETS.items():
        target = data[target_column]
        model_candidates = (
            [("random_forest", RandomForestRegressor(n_estimators=18, max_depth=14, min_samples_leaf=5, max_samples=0.5, n_jobs=4, random_state=42)),
             ("extra_trees", ExtraTreesRegressor(n_estimators=18, max_depth=14, min_samples_leaf=5, n_jobs=4, random_state=42))]
            if kind == "regression" else
            [("random_forest", RandomForestClassifier(n_estimators=18, max_depth=14, min_samples_leaf=5, max_samples=0.5, class_weight="balanced", n_jobs=4, random_state=42)),
             ("extra_trees", ExtraTreesClassifier(n_estimators=18, max_depth=14, min_samples_leaf=5, class_weight="balanced", n_jobs=4, random_state=42))]
        )
        candidates = []
        for model_name, estimator in model_candidates:
            pipeline = make_pipeline(kind, estimator)
            pipeline.fit(features.iloc[train_x], target.iloc[train_x])
            valid_prediction = pipeline.predict(features.iloc[valid_x])
            if kind == "regression":
                score = regression_metrics(target.iloc[valid_x], valid_prediction)["mae"]
            else:
                score = f1_score(target.iloc[valid_x], valid_prediction, zero_division=0)
            candidates.append((score, model_name, pipeline))
        selected = (min(candidates, key=lambda item: item[0]) if kind == "regression" else max(candidates, key=lambda item: item[0]))
        score, model_name, pipeline = selected
        selected_estimator = clone(pipeline)
        selected_estimator.set_params(model__n_estimators=30, model__max_depth=16, model__min_samples_leaf=4, model__n_jobs=4)
        full_train_x = np.concatenate((train_x, valid_x))
        selected_estimator.fit(features.iloc[full_train_x], target.iloc[full_train_x])
        pipeline = selected_estimator
        test_prediction = pipeline.predict(features.iloc[test_x])
        if kind == "regression":
            final_metrics = regression_metrics(target.iloc[test_x], test_prediction)
        else:
            final_metrics = {
                "accuracy": float(accuracy_score(target.iloc[test_x], test_prediction)),
                "precision": float(precision_score(target.iloc[test_x], test_prediction, zero_division=0)),
                "recall": float(recall_score(target.iloc[test_x], test_prediction, zero_division=0)),
                "f1": float(f1_score(target.iloc[test_x], test_prediction, zero_division=0)),
            }
            probabilities = pipeline.predict_proba(features.iloc[test_x])[:, 1]
            if target.iloc[test_x].nunique() == 2:
                final_metrics["roc_auc"] = float(roc_auc_score(target.iloc[test_x], probabilities))
        artifact = {"pipeline": pipeline, "kind": kind, "target": target_column, "selected_algorithm": model_name, "validation_score": float(score), "features_used": MODEL_FEATURES, "features_excluded": EXCLUDED_FEATURES, "version": version}
        filename = f"{key}_model.pkl"
        import joblib
        joblib.dump(artifact, output_dir / filename, compress=3)
        report["models"][key] = {"file": filename, "target": target_column, "kind": kind, "selected_algorithm": model_name, "validation_metric": "mae" if kind == "regression" else "f1", "validation_score": float(score), "test_metrics": final_metrics}
        print(f"Trained {key}: {model_name}; test={final_metrics}")

    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        Base.metadata.create_all(engine)
        with SessionLocal.begin() as session:
            for model_name, model_report in report["models"].items():
                session.add(ModelVersion(model_name=model_name, version=version, metrics=model_report, feature_names=MODEL_FEATURES))
        report["mysql_model_version_sync"] = "saved"
    except SQLAlchemyError:
        report["mysql_model_version_sync"] = "unavailable; model files and measured metrics were saved locally"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved model metrics to {metrics_path}")


if __name__ == "__main__":
    main()
