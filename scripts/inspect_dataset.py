import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import settings

CATEGORICAL = [
    "disruption_type", "industry", "supplier_tier", "supplier_region", "supplier_size",
    "has_backup_supplier", "response_type", "permanent_supplier_change",
]
NUMERICAL = [
    "disruption_severity", "production_impact_pct", "revenue_loss_usd", "response_time_days",
    "partial_recovery_days", "full_recovery_days",
]


def main() -> None:
    path = settings.path(settings.dataset_path)
    frame = pd.read_csv(path)
    numeric = frame.select_dtypes(include="number")
    numeric_profile = {}
    for column in NUMERICAL:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        numeric_profile[column] = {
            "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
            "median": float(values.median()), "q1": float(q1), "q3": float(q3),
            "iqr_outlier_count": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()),
        }
    profile = {
        "source_file": path.relative_to(settings.path(".")).as_posix(),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values": {column: int(count) for column, count in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_disruption_ids": int(frame["disruption_id"].duplicated().sum()),
        "categorical_values": {column: {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()} for column in CATEGORICAL},
        "numerical_ranges_and_iqr_outliers": numeric_profile,
        "valid_rows_for_required_schema": int(len(frame.dropna(subset=list(frame.columns)))),
    }
    target = settings.path("evaluation/dataset_profile.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps({"profile": str(target), "row_count": profile["row_count"], "column_count": profile["column_count"], "duplicate_ids": profile["duplicate_disruption_ids"]}, indent=2))


if __name__ == "__main__":
    main()
