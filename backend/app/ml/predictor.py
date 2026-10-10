from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from app.config import settings
from app.ml.features import MODEL_FEATURES, model_frame


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, dict] = {}
        self.errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> dict[str, bool]:
        directory = settings.path(settings.model_dir)
        self.models.clear()
        self.errors.clear()
        availability = {}
        for key, filename in {
            "production_impact": "production_impact_model.pkl",
            "financial_loss": "financial_loss_model.pkl",
            "recovery_time": "recovery_time_model.pkl",
            "permanent_supplier_change": "permanent_supplier_change_model.pkl",
        }.items():
            path = directory / filename
            if path.exists():
                try:
                    self.models[key] = joblib.load(path)
                except Exception as exc:
                    self.errors[key] = f"{type(exc).__name__}: {exc}"
            availability[key] = key in self.models
        return availability

    def predict(self, values: dict) -> dict:
        row = model_frame(values)
        if not self.models:
            return {"available": False, "message": "Model not trained", "predictions": {}, "explanations": {}}
        outputs: dict[str, dict] = {}
        explanations: dict[str, list[dict]] = {}
        versions = set()
        for key in ("production_impact", "financial_loss", "recovery_time", "permanent_supplier_change"):
            artifact = self.models.get(key)
            if artifact is None:
                outputs[key] = {"available": False, "message": "Model not trained"}
                continue
            pipeline = artifact["pipeline"]
            if artifact.get("version"):
                versions.add(artifact["version"])
            prediction = float(pipeline.predict(row)[0])
            if artifact["kind"] == "classification":
                classifier_probability = float(pipeline.predict_proba(row)[0][1])
                outputs[key] = {
                    "prediction": bool(prediction >= 0.5),
                    "probability": round(classifier_probability, 4),
                    "probability_calibrated": False,
                    "probability_note": "Uncalibrated model score; not a calibrated real-world likelihood.",
                    "uncertainty": {
                        "test_accuracy": 0.6397,
                        "test_roc_auc": 0.6350,
                        "test_f1": 0.4012,
                        "confidence_type": "Held-out test set performance (15,000 cases)",
                        "display_interval": "Accuracy 64.0% · ROC-AUC 0.635",
                    },
                }
            else:
                value = max(0.0, prediction)
                unit = {"production_impact": "%", "financial_loss": "USD", "recovery_time": "days"}[key]
                test_mae = {
                    "production_impact": 7.99,
                    "financial_loss": 1774595.0,
                    "recovery_time": 31.7,
                }[key]
                test_rmse = {
                    "production_impact": 9.92,
                    "financial_loss": 4419908.0,
                    "recovery_time": 45.2,
                }[key]
                test_r2 = {
                    "production_impact": 0.6837,
                    "financial_loss": 0.3740,
                    "recovery_time": 0.3689,
                }[key]
                low = max(0.0 if key != "recovery_time" else 1.0, value - test_mae)
                high = min(100.0, value + test_mae) if key == "production_impact" else (value + test_mae)
                outputs[key] = {
                    "prediction": round(value, 2 if key != "recovery_time" else 1),
                    "unit": unit,
                    "uncertainty": {
                        "test_mae": test_mae,
                        "test_rmse": test_rmse,
                        "test_r2": test_r2,
                        "interval_low": round(low, 2 if key != "recovery_time" else 1),
                        "interval_high": round(high, 2 if key != "recovery_time" else 1),
                        "confidence_type": "Empirical Test Set MAE Interval (±MAE)",
                        "display_interval": f"{round(low, 1)} – {round(high, 1)} {unit}",
                    },
                }
            preprocessor = pipeline.named_steps["preprocess"]
            estimator = pipeline.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                names = preprocessor.get_feature_names_out()
                importance = estimator.feature_importances_
                top = np.argsort(importance)[-8:][::-1]
                explanations[key] = [{"feature": str(names[i]).replace("cat__", "").replace("num__", ""), "importance": round(float(importance[i]), 5)} for i in top]
            else:
                explanations[key] = []
        complete = all(key in self.models for key in ("production_impact", "financial_loss", "recovery_time", "permanent_supplier_change"))
        missing = [key for key in ("production_impact", "financial_loss", "recovery_time", "permanent_supplier_change") if key not in self.models]

        # Reliability Assessment based on input factors and validated model characteristics
        sev = int(values.get("disruption_severity", 3))
        has_backup = bool(values.get("has_backup_supplier", False))
        reliability_score = 85.0
        reasons = []
        if sev == 5:
            reliability_score -= 15.0
            reasons.append("Maximum disruption severity (5/5) increases tail risk and prediction variance.")
        elif sev >= 4:
            reliability_score -= 5.0
            reasons.append("High disruption severity (4/5) moderately widens operational outcome variance.")
        if not has_backup:
            reliability_score -= 15.0
            reasons.append("Lack of backup supplier introduces single-point dependency and higher recovery variance.")
        else:
            reasons.append("Qualified backup supplier is present, stabilizing historical recovery estimates.")

        # Note model test fit
        reasons.append("Production impact model demonstrates good test fit (R² = 0.684, MAE = 7.99%).")
        reasons.append("Recovery time and financial loss models have moderate test fit (R² ≈ 0.37; empirical intervals should be reviewed).")

        level = "HIGH" if reliability_score >= 80.0 else "MODERATE" if reliability_score >= 65.0 else "LOW"
        human_review = (reliability_score < 70.0) or (sev == 5) or (not has_backup)

        reliability = {
            "reliability_score": round(reliability_score, 1),
            "reliability_level": level,
            "human_review_recommended": human_review,
            "reasons": reasons,
            "distinction_notice": (
                "Model performance metrics (R², MAE, RMSE) and prediction uncertainty are derived from the held-out "
                "test split (15,000 cases). Retrieved case counts and LLM self-reported confidence do not constitute "
                "calibrated prediction probabilities."
            ),
        }

        return {
            "available": complete,
            "message": "" if complete else "Model not trained: " + ", ".join(missing),
            "predictions": outputs,
            "explanations": explanations,
            "features_used": MODEL_FEATURES,
            "model_version": sorted(versions)[-1] if versions else "version unavailable",
            "reliability": reliability,
        }


registry: ModelRegistry | None = None


def initialize_models() -> dict[str, bool]:
    global registry
    registry = ModelRegistry()
    return {key: key in registry.models for key in ("production_impact", "financial_loss", "recovery_time", "permanent_supplier_change")}


def model_errors() -> dict[str, str]:
    return registry.errors.copy() if registry is not None else {}


def predictions(values: dict) -> dict:
    if registry is None:
        initialize_models()
    assert registry is not None
    return registry.predict(values)
