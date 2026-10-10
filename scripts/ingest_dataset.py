from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import Disruption

EXPECTED_COLUMNS = [
    "disruption_id", "disruption_type", "industry", "supplier_tier", "supplier_region", "supplier_size",
    "has_backup_supplier", "disruption_severity", "production_impact_pct", "revenue_loss_usd",
    "response_type", "response_time_days", "partial_recovery_days", "full_recovery_days", "permanent_supplier_change",
]
NUMERIC = ["disruption_severity", "production_impact_pct", "revenue_loss_usd", "response_time_days", "partial_recovery_days", "full_recovery_days"]
BOOLEAN = ["has_backup_supplier", "permanent_supplier_change"]
TEXT = ["disruption_id", "disruption_type", "industry", "supplier_tier", "supplier_region", "supplier_size", "response_type"]


def parse_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_dataset() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(settings.path(settings.dataset_path))
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    unexpected_columns = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or unexpected_columns:
        raise ValueError(f"CSV schema mismatch; missing={missing_columns}, unexpected={unexpected_columns}")
    frame = frame[EXPECTED_COLUMNS].copy()
    for column in TEXT:
        frame[column] = frame[column].astype("string").str.strip()
        frame[column] = frame[column].replace({"": pd.NA})
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN:
        frame[column] = frame[column].map(parse_bool)
    before = len(frame)
    required = EXPECTED_COLUMNS
    frame = frame.dropna(subset=required).copy()
    dropped_invalid = before - len(frame)
    exact_duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()
    duplicate_ids = int(frame.duplicated(subset=["disruption_id"], keep="first").sum())
    frame = frame.drop_duplicates(subset=["disruption_id"], keep="first").copy()
    if ((frame["disruption_severity"] < 1) | (frame["disruption_severity"] > 5)).any():
        raise ValueError("disruption_severity values must be between 1 and 5")
    if ((frame["production_impact_pct"] < 0) | (frame["production_impact_pct"] > 100)).any():
        raise ValueError("production_impact_pct values must be between 0 and 100")
    nonnegative = ["revenue_loss_usd", "response_time_days", "partial_recovery_days", "full_recovery_days"]
    if (frame[nonnegative] < 0).any().any():
        raise ValueError("loss and duration columns cannot be negative")
    stats = {"source_rows": before, "valid_rows": len(frame), "dropped_invalid_rows": dropped_invalid, "exact_duplicate_rows_removed": exact_duplicates, "duplicate_disruption_ids_removed": duplicate_ids}
    return frame, stats


def main() -> None:
    frame, stats = load_dataset()
    Base.metadata.create_all(engine)
    inserted = 0
    already_present = 0
    with SessionLocal.begin() as session:
        for offset in range(0, len(frame), 5000):
            chunk = frame.iloc[offset:offset + 5000]
            ids = chunk["disruption_id"].astype(str).tolist()
            existing = set(session.scalars(select(Disruption.disruption_id).where(Disruption.disruption_id.in_(ids))).all())
            records = []
            for row in chunk.to_dict(orient="records"):
                identifier = str(row["disruption_id"])
                if identifier in existing:
                    already_present += 1
                    continue
                row["disruption_id"] = identifier
                row["supplier_tier"] = str(row["supplier_tier"])
                row["disruption_severity"] = int(row["disruption_severity"])
                row["production_impact_pct"] = float(row["production_impact_pct"])
                row["revenue_loss_usd"] = float(row["revenue_loss_usd"])
                for column in ("response_time_days", "partial_recovery_days", "full_recovery_days"):
                    row[column] = int(row[column])
                row["has_backup_supplier"] = bool(row["has_backup_supplier"])
                row["permanent_supplier_change"] = bool(row["permanent_supplier_change"])
                records.append(row)
            if records:
                session.bulk_insert_mappings(Disruption, records)
                inserted += len(records)
    report = {**stats, "inserted": inserted, "already_present": already_present}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
