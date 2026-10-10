from typing import Any

import pandas as pd

CATEGORICAL_FEATURES = [
    "disruption_type", "industry", "supplier_tier", "supplier_region", "supplier_size", "has_backup_supplier",
]
NUMERICAL_FEATURES = ["disruption_severity"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGETS = {
    "production_impact": ("production_impact_pct", "regression"),
    "financial_loss": ("revenue_loss_usd", "regression"),
    "recovery_time": ("full_recovery_days", "regression"),
    "permanent_supplier_change": ("permanent_supplier_change", "classification"),
}
EXCLUDED_FEATURES = {
    "production_impact_pct": "prediction target",
    "revenue_loss_usd": "prediction target",
    "full_recovery_days": "prediction target",
    "permanent_supplier_change": "prediction target",
    "response_type": "strategy is selected after initial event assessment and would leak response decisions",
    "response_time_days": "response outcome is not known at initial prediction time",
    "partial_recovery_days": "later recovery outcome; would leak future information into full recovery prediction",
}


def model_frame(values: dict[str, Any]) -> pd.DataFrame:
    row = {key: values[key] for key in MODEL_FEATURES}
    row["supplier_tier"] = str(row["supplier_tier"])
    row["has_backup_supplier"] = str(bool(row["has_backup_supplier"]))
    return pd.DataFrame([row], columns=MODEL_FEATURES)
