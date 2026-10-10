from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Disruption

logger = logging.getLogger("uvicorn.error")

_SUPPLIER_RISK_CACHE: list[dict[str, Any]] | None = None


def _load_disruption_records(db: Session | None = None) -> list[dict[str, Any]]:
    """
    Load disruption records from MySQL if available, or fall back to dataset CSV.
    """
    records: list[dict[str, Any]] = []
    if db is not None:
        try:
            stmt = select(
                Disruption.disruption_id,
                Disruption.disruption_type,
                Disruption.industry,
                Disruption.supplier_tier,
                Disruption.supplier_region,
                Disruption.supplier_size,
                Disruption.has_backup_supplier,
                Disruption.disruption_severity,
                Disruption.production_impact_pct,
                Disruption.revenue_loss_usd,
                Disruption.full_recovery_days,
                Disruption.permanent_supplier_change,
            ).limit(20000)
            rows = db.execute(stmt).all()
            if rows:
                for r in rows:
                    records.append({
                        "disruption_id": r.disruption_id,
                        "disruption_type": r.disruption_type,
                        "industry": r.industry,
                        "supplier_tier": str(r.supplier_tier),
                        "supplier_region": r.supplier_region,
                        "supplier_size": r.supplier_size,
                        "has_backup_supplier": bool(r.has_backup_supplier),
                        "disruption_severity": int(r.disruption_severity),
                        "production_impact_pct": float(r.production_impact_pct or 0.0),
                        "revenue_loss_usd": float(r.revenue_loss_usd or 0.0),
                        "full_recovery_days": float(r.full_recovery_days or 0.0),
                        "permanent_supplier_change": bool(r.permanent_supplier_change),
                    })
                return records
        except Exception as exc:
            logger.warning("Could not query disruptions from database for supplier risk: %s", exc)

    # Fallback to dataset CSV
    csv_path = settings.path(settings.dataset_path)
    if csv_path.exists():
        try:
            cols = [
                "disruption_id", "disruption_type", "industry", "supplier_tier",
                "supplier_region", "supplier_size", "has_backup_supplier",
                "disruption_severity", "production_impact_pct", "revenue_loss_usd",
                "full_recovery_days", "permanent_supplier_change",
            ]
            df = pd.read_csv(csv_path, usecols=cols, nrows=25000)
            for _, r in df.iterrows():
                records.append({
                    "disruption_id": str(r["disruption_id"]),
                    "disruption_type": str(r["disruption_type"]),
                    "industry": str(r["industry"]),
                    "supplier_tier": str(r["supplier_tier"]),
                    "supplier_region": str(r["supplier_region"]),
                    "supplier_size": str(r["supplier_size"]),
                    "has_backup_supplier": bool(r["has_backup_supplier"]),
                    "disruption_severity": int(r["disruption_severity"]),
                    "production_impact_pct": float(r["production_impact_pct"]),
                    "revenue_loss_usd": float(r["revenue_loss_usd"]),
                    "full_recovery_days": float(r["full_recovery_days"]),
                    "permanent_supplier_change": bool(r["permanent_supplier_change"]),
                })
        except Exception as exc:
            logger.warning("Could not read dataset CSV for supplier risk: %s", exc)

    return records


def get_supplier_risk_overview(
    db: Session | None = None,
    region: str | None = None,
    tier: str | None = None,
    industry: str | None = None,
    min_severity: int | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Computes operational supplier risk profiles across segments using historical records.
    Explicitly labels metrics as historical operational vulnerability indicators (not real-time).
    """
    raw_records = _load_disruption_records(db)
    if not raw_records:
        return {
            "notice": "HISTORICAL OPERATIONAL VULNERABILITY: Derived strictly from past disruption records. Not real-time telemetry.",
            "total_records_analyzed": 0,
            "segments": [],
            "summary": {
                "critical_count": 0,
                "elevated_count": 0,
                "moderate_count": 0,
                "low_count": 0,
                "avg_recovery_days": 0.0,
                "backup_coverage_pct": 0.0,
            },
            "filters_applied": {
                "region": region, "tier": tier, "industry": industry, "min_severity": min_severity,
            },
        }

    # Apply filters
    filtered = raw_records
    if region and region.strip():
        filtered = [r for r in filtered if r["supplier_region"].lower() == region.strip().lower()]
    if tier and tier.strip():
        clean_tier = tier.strip().lower().replace("tier", "").strip()
        filtered = [r for r in filtered if str(r["supplier_tier"]).lower().replace("tier", "").strip() == clean_tier]
    if industry and industry.strip():
        filtered = [r for r in filtered if r["industry"].lower() == industry.strip().lower()]
    if min_severity is not None:
        filtered = [r for r in filtered if r["disruption_severity"] >= min_severity]

    # Aggregate by (tier, region, industry)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in filtered:
        key = (str(r["supplier_tier"]).replace("Tier", "").strip(), r["supplier_region"], r["industry"])
        groups[key].append(r)

    segment_results: list[dict[str, Any]] = []

    for (t, reg, ind), rows in groups.items():
        count = len(rows)
        avg_sev = sum(r["disruption_severity"] for r in rows) / count
        avg_rec = sum(r["full_recovery_days"] for r in rows) / count
        rec_series = [r["full_recovery_days"] for r in rows]
        med_rec = round(float(pd.Series(rec_series).median()), 1) if rec_series else round(avg_rec, 1)
        avg_imp = sum(r["production_impact_pct"] for r in rows) / count
        avg_loss = sum(r["revenue_loss_usd"] for r in rows) / count
        no_backup_count = sum(1 for r in rows if not r["has_backup_supplier"])
        no_backup_pct = round((no_backup_count / count) * 100.0, 1)
        backup_avail_pct = round(100.0 - no_backup_pct, 1)
        perm_change_count = sum(1 for r in rows if r["permanent_supplier_change"])
        perm_change_pct = round((perm_change_count / count) * 100.0, 1)

        # Composite score:
        # Severity (25%), Recovery days / 120 (25%), No backup (30%), Perm change (20%)
        sev_norm = (avg_sev / 5.0) * 100.0
        rec_norm = min(100.0, (avg_rec / 120.0) * 100.0)
        raw_score = 0.25 * sev_norm + 0.25 * rec_norm + 0.30 * no_backup_pct + 0.20 * perm_change_pct
        composite_score = round(min(100.0, max(0.0, raw_score)), 1)

        if composite_score >= 70.0:
            category = "Critical Vulnerability"
            badge_tone = "red"
        elif composite_score >= 50.0:
            category = "Elevated Risk"
            badge_tone = "amber"
        elif composite_score >= 35.0:
            category = "Moderate Vulnerability"
            badge_tone = "blue"
        else:
            category = "Low Historical Risk"
            badge_tone = "green"

        reasons = []
        if no_backup_pct >= 60.0:
            reasons.append(f"{no_backup_pct}% of recorded incidents lacked a qualified backup supplier.")
        if avg_rec >= 70.0:
            reasons.append(f"Prolonged average recovery duration ({round(avg_rec, 1)} days; median {med_rec} days).")
        if perm_change_pct >= 30.0:
            reasons.append(f"Frequent permanent supplier replacement ({perm_change_pct}% of cases).")
        if avg_sev >= 3.8:
            reasons.append(f"High average disruption severity ({round(avg_sev, 1)}/5).")
        if not reasons:
            reasons.append(f"Stable profile with {backup_avail_pct}% backup coverage and {round(avg_rec, 1)}d avg recovery.")

        segment_results.append({
            "segment_id": f"Tier {t} · {reg} · {ind}",
            "supplier_tier": f"Tier {t}",
            "tier_number": t,
            "supplier_region": reg,
            "industry": ind,
            "disruption_count": count,
            "historical_incident_count": count,
            "composite_risk_score": composite_score,
            "risk_category": category,
            "badge_tone": badge_tone,
            "average_severity": round(avg_sev, 2),
            "avg_severity": round(avg_sev, 2),
            "average_recovery_days": round(avg_rec, 1),
            "median_recovery_days": med_rec,
            "average_production_impact_pct": round(avg_imp, 1),
            "average_revenue_loss_usd": round(avg_loss, 2),
            "no_backup_supplier_pct": no_backup_pct,
            "backup_availability_pct": backup_avail_pct,
            "permanent_supplier_change_rate_pct": perm_change_pct,
            "vulnerability_reasons": reasons,
            "reasons": reasons,
            "supporting_disruption_ids": [r["disruption_id"] for r in rows[:6]],
        })

    # Sort by highest composite risk score first
    segment_results.sort(key=lambda x: x["composite_risk_score"], reverse=True)

    critical_count = sum(1 for s in segment_results if s["risk_category"] == "Critical Vulnerability")
    elevated_count = sum(1 for s in segment_results if s["risk_category"] == "Elevated Risk")
    moderate_count = sum(1 for s in segment_results if s["risk_category"] == "Moderate Vulnerability")
    low_count = sum(1 for s in segment_results if s["risk_category"] == "Low Historical Risk")

    total_disruptions = len(filtered)
    overall_avg_rec = round(sum(r["full_recovery_days"] for r in filtered) / max(1, total_disruptions), 1)
    backup_count = sum(1 for r in filtered if r["has_backup_supplier"])
    backup_coverage = round((backup_count / max(1, total_disruptions)) * 100.0, 1)

    return {
        "notice": "HISTORICAL OPERATIONAL VULNERABILITY: Calculated from historical disruption incident records. Not real-time sensor or live telemetry data.",
        "data_scope": "Historical Disruption Incident Records",
        "total_records_analyzed": total_disruptions,
        "total_disruptions_analyzed": total_disruptions,
        "total_segments_identified": len(segment_results),
        "critical_segments_count": critical_count,
        "elevated_segments_count": elevated_count,
        "moderate_segments_count": moderate_count,
        "low_segments_count": low_count,
        "avg_historical_recovery_days": overall_avg_rec,
        "avg_backup_coverage_pct": backup_coverage,
        "summary": {
            "critical_count": critical_count,
            "elevated_count": elevated_count,
            "moderate_count": moderate_count,
            "low_count": low_count,
            "avg_recovery_days": overall_avg_rec,
            "backup_coverage_pct": backup_coverage,
        },
        "filters_applied": {
            "region": region, "tier": tier, "industry": industry, "min_severity": min_severity,
        },
        "segments": segment_results[:limit],
    }
