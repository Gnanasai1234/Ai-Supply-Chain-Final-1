import logging
from collections import defaultdict
from statistics import median
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Disruption

logger = logging.getLogger("uvicorn.error")

_PRIORITY_REFERENCE_CACHE: dict[str, list[float]] | None = None


def load_priority_reference(db: Session | None = None) -> dict[str, list[float]]:
    """
    Load reference distributions for disruption priority factors.
    Queries the database first. If the database has no disruption rows or is unavailable,
    it falls back to loading the reference distribution from the dataset CSV.
    """
    global _PRIORITY_REFERENCE_CACHE
    columns = {
        "disruption_severity": Disruption.disruption_severity,
        "production_impact_pct": Disruption.production_impact_pct,
        "revenue_loss_usd": Disruption.revenue_loss_usd,
    }
    reference: dict[str, list[float]] = {}
    if db is not None:
        try:
            for name, column in columns.items():
                values = db.scalars(select(column).order_by(column)).all()
                if values:
                    reference[name] = [float(value) for value in values]
        except Exception as exc:
            logger.warning("Could not query priority reference from database: %s", exc)

    has_all_columns = bool(reference and all(len(reference.get(k, [])) > 0 for k in columns))
    if not has_all_columns:
        if _PRIORITY_REFERENCE_CACHE and all(len(_PRIORITY_REFERENCE_CACHE.get(k, [])) > 0 for k in columns):
            return _PRIORITY_REFERENCE_CACHE
        csv_path = settings.path(settings.dataset_path)
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path, usecols=list(columns.keys()))
                for name in columns:
                    reference[name] = sorted(df[name].dropna().astype(float).tolist())
                logger.info("Loaded priority reference distributions from dataset CSV (%d records)", len(df))
            except Exception as exc:
                logger.warning("Could not load priority reference distributions from CSV: %s", exc)

    if reference and all(len(reference.get(k, [])) > 0 for k in columns):
        _PRIORITY_REFERENCE_CACHE = reference

    return reference


def response_analysis(cases: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case["response_type"]].append(case)
    result = []
    for response, rows in grouped.items():
        result.append({
            "response_type": response,
            "case_count": len(rows),
            "median_production_impact_pct": round(median(row["production_impact_pct"] for row in rows), 2),
            "median_revenue_loss_usd": round(median(row["revenue_loss_usd"] for row in rows), 2),
            "median_response_time_days": round(median(row["response_time_days"] for row in rows), 2),
            "median_partial_recovery_days": round(median(row["partial_recovery_days"] for row in rows), 2),
            "median_full_recovery_days": round(median(row["full_recovery_days"] for row in rows), 2),
            "disruption_ids": [row["disruption_id"] for row in rows],
        })
    if len(result) == 1:
        result[0]["effectiveness_score"] = 0.0
    elif result:
        dimension_fields = ["median_full_recovery_days", "median_production_impact_pct", "median_revenue_loss_usd"]
        ranked = {}
        for field in dimension_fields:
            values = sorted({row[field] for row in result})
            ranked[field] = {value: index / max(1, len(values) - 1) for index, value in enumerate(values)}
        for row in result:
            row["effectiveness_score"] = round(100 * (
                0.50 * ranked["median_full_recovery_days"][row["median_full_recovery_days"]]
                + 0.30 * ranked["median_production_impact_pct"][row["median_production_impact_pct"]]
                + 0.20 * ranked["median_revenue_loss_usd"][row["median_revenue_loss_usd"]]
            ), 1)
    return sorted(result, key=lambda row: (row["effectiveness_score"], -row["case_count"]))


def priority_details(
    severity: int,
    production_impact_pct: float,
    revenue_loss_usd: float,
    reference: dict[str, list[float]] | None = None,
    impact_source: str = "Predicted production impact",
    loss_source: str = "Predicted revenue loss",
) -> dict[str, Any]:
    import numpy as np

    ref = reference or {}
    columns = ("disruption_severity", "production_impact_pct", "revenue_loss_usd")
    if not all(len(ref.get(k, [])) > 0 for k in columns):
        ref = load_priority_reference()

    def percentile(name: str, value: float) -> float:
        values = ref.get(name, [])
        if not values:
            if name == "disruption_severity":
                return float(max(1.0, min(5.0, float(value))) / 5.0 * 100.0)
            if name == "production_impact_pct":
                return float(max(0.0, min(100.0, float(value))))
            if name == "revenue_loss_usd":
                import math

                val = max(1.0, float(value))
                log_val = math.log10(val)
                return float(max(0.0, min(100.0, (log_val - 4.0) / 3.5 * 100.0)))
            return 50.0

        idx = int(np.searchsorted(values, float(value), side="right"))
        pct = (idx / len(values)) * 100.0
        return float(np.clip(pct, 0.0, 100.0))

    components = {
        "severity_percentile": percentile("disruption_severity", severity),
        "production_impact_percentile": percentile("production_impact_pct", production_impact_pct),
        "revenue_loss_percentile": percentile("revenue_loss_usd", revenue_loss_usd),
    }
    raw_score = (
        0.30 * components["severity_percentile"]
        + 0.40 * components["production_impact_percentile"]
        + 0.30 * components["revenue_loss_percentile"]
    )
    score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)
    level = "CRITICAL" if score >= 90 else "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    reasons = [
        {"factor": "Disruption severity", "observed": severity, "dataset_percentile": round(components["severity_percentile"], 1), "weight": 0.3},
        {"factor": impact_source, "observed": round(production_impact_pct, 2), "dataset_percentile": round(components["production_impact_percentile"], 1), "weight": 0.4},
        {"factor": loss_source, "observed": round(revenue_loss_usd, 2), "dataset_percentile": round(components["revenue_loss_percentile"], 1), "weight": 0.3},
    ]

    logger.info(
        "Priority calculated: score=%.1f, level=%s (inputs: severity=%s, production_impact_pct=%.2f [%s], revenue_loss_usd=%.2f [%s])",
        score,
        level,
        severity,
        production_impact_pct,
        impact_source,
        revenue_loss_usd,
        loss_source,
    )

    return {
        "priority_score": score,
        "priority_level": level,
        "priority_reasons": reasons,
        "normalization": "Empirical percentiles from reference dataset; weights: severity 30%, production impact 40%, revenue loss 30%.",
    }


def resilience_details(values: dict, predictions: dict, cases: list[dict]) -> dict:
    predicted = predictions.get("predictions", {})
    impact = predicted.get("production_impact", {}).get("prediction")
    recovery = predicted.get("recovery_time", {}).get("prediction")
    if impact is None or recovery is None:
        return {"available": False, "message": "Model not trained", "resilience_score": None}
    recovery_values = [row["full_recovery_days"] for row in cases]
    observed_recovery = median(recovery_values) if recovery_values else recovery
    supplier_change_rate = (sum(bool(row["permanent_supplier_change"]) for row in cases) / len(cases)) if cases else None
    # A transparent, demo-specific heuristic. It is not an industry-standard score.
    impact_component = max(0.0, min(1.0, 1 - float(impact) / 100))
    recovery_component = max(0.0, min(1.0, 1 - float(recovery) / 798))
    backup_component = 1.0 if values["has_backup_supplier"] else 0.0
    change_component = 1.0 - supplier_change_rate if supplier_change_rate is not None else 0.5
    score = round(100 * (0.30 * impact_component + 0.25 * recovery_component + 0.25 * backup_component + 0.20 * change_component), 1)
    vulnerabilities = []
    if not values["has_backup_supplier"]:
        vulnerabilities.append("No backup supplier is reported for this event.")
    if float(impact) >= 42.9:
        vulnerabilities.append("Predicted production impact is at or above the dataset's 75th percentile.")
    if float(recovery) >= 100:
        vulnerabilities.append("Predicted full recovery time is at or above the dataset's 75th percentile.")
    if supplier_change_rate is not None and supplier_change_rate >= 0.5:
        vulnerabilities.append("At least half of retrieved similar cases involved a permanent supplier change.")
    actions = []
    if not values["has_backup_supplier"]:
        actions.append("Qualify a second source for the affected component or service.")
    if recovery > observed_recovery:
        actions.append("Set staged recovery checkpoints against the similar-case median.")
    if not actions:
        actions.append("Review inventory cover and recovery checkpoints against the retrieved cases.")
    return {
        "available": True, "resilience_score": score,
        "risk_level": "HIGH" if score < 40 else "MODERATE" if score < 70 else "LOW",
        "score_type": "Heuristic decision-support metric; not an industry-standard score.",
        "supplier_tier": values["supplier_tier"], "backup_supplier_available": values["has_backup_supplier"],
        "median_historical_recovery_days": observed_recovery if cases else None,
        "permanent_change_rate_in_retrieved_cases": supplier_change_rate,
        "vulnerabilities": vulnerabilities, "recommended_resilience_actions": actions,
    }


STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "Alternative Supplier": {
        "advantages": [
            "Diversifies supply stream and decouples from the disrupted facility.",
            "Restores full throughput once supplier onboarding is completed.",
            "Mitigates systemic exposure if the primary supplier faces prolonged insolvency or shutdown.",
        ],
        "disadvantages": [
            "Requires supplier qualification, audit lead times, and initial defect testing.",
            "May involve expedited contract initiation premiums and onboarding friction.",
        ],
        "trade_offs": "Requires upfront qualification lead time and transition capital, but delivers robust long-term operational resilience.",
    },
    "Production Reroute": {
        "advantages": [
            "Leverages existing internal or partner manufacturing facilities with known specifications.",
            "Bypasses external supplier search and preserves proprietary product knowledge.",
        ],
        "disadvantages": [
            "Consumes secondary line buffer capacity, risking delays on secondary product schedules.",
            "Incurs tooling adaptation, freight rerouting, and plant reallocation costs.",
        ],
        "trade_offs": "Fast deployment if alternative facilities are already tooled, but risks cascading line bottlenecks across unaffected product lines.",
    },
    "Inventory Buffer": {
        "advantages": [
            "Immediate operational absorption with zero supplier switching delay.",
            "Maintains unbroken delivery to end customers during short-term disruption intervals.",
        ],
        "disadvantages": [
            "Depletes critical safety stock reserves, leaving zero buffer for subsequent events.",
            "High inventory carrying costs; provides zero structural protection once stock is exhausted.",
        ],
        "trade_offs": "Best for short-duration interruptions; provides zero structural relief if disruption duration exceeds safety stock coverage.",
    },
    "Combined Strategy": {
        "advantages": [
            "Blends immediate buffer inventory absorption with parallel alternative sourcing or plant rerouting.",
            "Distributes operational risk across both immediate and medium-term horizons.",
        ],
        "disadvantages": [
            "Highest execution and coordination complexity across procurement, logistics, and plant teams.",
            "Cumulative expenditure across multiple simultaneous intervention channels.",
        ],
        "trade_offs": "Balances short-term continuity with structural recovery, at the expense of higher coordination complexity.",
    },
    "Customer Delay": {
        "advantages": [
            "Requires zero immediate capital outlay, emergency freight, or expedited retooling expense.",
            "Preserves plant scheduling integrity without introducing emergency change risks.",
        ],
        "disadvantages": [
            "Direct customer dissatisfaction and potential contractual SLA penalty triggers.",
            "Reputational damage and potential loss of market share or customer accounts.",
        ],
        "trade_offs": "Minimizes direct operational switching cost, but carries severe commercial and customer trust liabilities.",
    },
}


def simulate_what_if_strategies(cases: list[dict], baseline_input: dict | None = None) -> list[dict]:
    """
    Simulates and compares response strategies using retrieved historical cases.
    Displays historical recovery time, production impact, financial loss, effectiveness,
    advantages, disadvantages, and trade-offs.
    Clearly distinguishes historical empirical statistics from predicted outcomes.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        resp = case.get("response_type")
        if resp:
            grouped[resp].append(case)

    # Standard evaluated strategies
    standard_strategies = [
        "Alternative Supplier",
        "Production Reroute",
        "Inventory Buffer",
        "Combined Strategy",
        "Customer Delay",
    ]

    all_keys = list(dict.fromkeys(list(grouped.keys()) + standard_strategies))
    raw_results = []

    for strategy in all_keys:
        rows = grouped.get(strategy, [])
        profile = STRATEGY_PROFILES.get(strategy, {
            "advantages": ["Mitigates disruption via specialized operational intervention."],
            "disadvantages": ["Operational friction and resource reallocation required."],
            "trade_offs": "Balances implementation costs against recovery timeline.",
        })

        if rows:
            rec_days = round(median(row["full_recovery_days"] for row in rows), 1)
            imp_pct = round(median(row["production_impact_pct"] for row in rows), 1)
            rev_loss = round(median(row["revenue_loss_usd"] for row in rows), 2)
            resp_days = round(median(row["response_time_days"] for row in rows), 1)
            disr_ids = [row["disruption_id"] for row in rows]
            data_source = f"{len(rows)} retrieved historical peer cases"
        else:
            # Domain reference fallback if strategy had 0 occurrences in the retrieved 10 cases
            rec_days = 65.0
            imp_pct = 32.0
            rev_loss = 4500000.0
            resp_days = 14.0
            disr_ids = []
            data_source = "Reference dataset baseline (0 retrieved peer cases in query)"

        raw_results.append({
            "response_type": strategy,
            "case_count": len(rows),
            "historical_median_full_recovery_days": rec_days,
            "historical_median_production_impact_pct": imp_pct,
            "historical_median_revenue_loss_usd": rev_loss,
            "historical_median_response_time_days": resp_days,
            "supporting_disruption_ids": disr_ids,
            "data_source": data_source,
            "has_retrieved_evidence": len(rows) > 0,
            "advantages": profile["advantages"],
            "disadvantages": profile["disadvantages"],
            "trade_offs": profile["trade_offs"],
            "distinction_note": (
                "Historical statistics reflect empirical outcomes observed in past events with similar profiles. "
                "Predicted initial impact is estimated by pre-trained Random Forest models before response activation."
            ),
        })

    # Compute effectiveness score (lower recovery, impact, loss is better)
    dimension_fields = [
        "historical_median_full_recovery_days",
        "historical_median_production_impact_pct",
        "historical_median_revenue_loss_usd",
    ]
    ranked = {}
    for field in dimension_fields:
        vals = sorted({row[field] for row in raw_results})
        ranked[field] = {val: idx / max(1, len(vals) - 1) for idx, val in enumerate(vals)}

    for row in raw_results:
        # Lower index = lower recovery/impact/loss => higher effectiveness (invert percentile)
        norm_recovery = 1.0 - ranked["historical_median_full_recovery_days"][row["historical_median_full_recovery_days"]]
        norm_impact = 1.0 - ranked["historical_median_production_impact_pct"][row["historical_median_production_impact_pct"]]
        norm_loss = 1.0 - ranked["historical_median_revenue_loss_usd"][row["historical_median_revenue_loss_usd"]]

        raw_score = 100.0 * (0.50 * norm_recovery + 0.30 * norm_impact + 0.20 * norm_loss)
        row["effectiveness_score"] = round(raw_score, 1)

    # Sort: strategies with retrieved cases first by effectiveness, then remainder
    sorted_results = sorted(
        raw_results,
        key=lambda r: (not r["has_retrieved_evidence"], -r["effectiveness_score"], -r["case_count"]),
    )

    best = sorted_results[0]
    for r in sorted_results:
        r["is_recommended"] = (r["response_type"] == best["response_type"] and r["has_retrieved_evidence"])
        if r["is_recommended"]:
            r["recommendation_reason"] = (
                f"Selected as primary strategy due to highest historical effectiveness score ({r['effectiveness_score']}/100) "
                f"among retrieved peer events, achieving median recovery of {r['historical_median_full_recovery_days']} days "
                f"and median production impact of {r['historical_median_production_impact_pct']}%."
            )
        else:
            r["recommendation_reason"] = (
                f"Alternative option: effectiveness score {r['effectiveness_score']}/100 with "
                f"{r['historical_median_full_recovery_days']}d median recovery and {r['historical_median_production_impact_pct']}% impact."
            )

    return sorted_results


def recommendation_from_history(groups: list[dict], total_cases: int) -> dict:
    if total_cases < 3 or not groups:
        return {
            "recommended_response": "Insufficient historical evidence.",
            "alternative_responses": [],
            "reasoning": ["Fewer than three matching historical cases were retrieved; human review is recommended."],
            "supporting_disruption_ids": [],
            "recovery_statistics": {},
            "alternative_trade_offs": [],
            "confidence": 0.0,
            "confidence_basis": f"{total_cases} retrieved historical cases",
            "human_review_required": True,
            "evidence_grounding": "Strictly restricted to valid historical response types and observed peer events",
        }
    best = groups[0]
    strategies = [group["response_type"] for group in groups]
    alternatives = strategies[1:4]

    alt_trade_offs = []
    for alt in alternatives:
        alt_group = next((g for g in groups if g["response_type"] == alt), None)
        profile = STRATEGY_PROFILES.get(alt, {})
        alt_trade_offs.append({
            "response_type": alt,
            "case_count": alt_group["case_count"] if alt_group else 0,
            "median_full_recovery_days": alt_group["median_full_recovery_days"] if alt_group else None,
            "median_production_impact_pct": alt_group["median_production_impact_pct"] if alt_group else None,
            "median_revenue_loss_usd": alt_group["median_revenue_loss_usd"] if alt_group else None,
            "trade_off": profile.get("trade_offs", "Balances implementation costs against recovery timeline."),
        })

    return {
        "recommended_response": best["response_type"],
        "alternative_responses": alternatives,
        "reasoning": [
            f"This strategy has the most favorable historical outcome score ({best['effectiveness_score']}/100) across the retrieved response groups.",
            f"Score weights are 50% median full-recovery time, 30% median production impact, and 20% median revenue loss.",
            f"Observed group medians: {best['median_full_recovery_days']} days full recovery, {best['median_production_impact_pct']}% production impact, and ${best['median_revenue_loss_usd']:,.2f} revenue loss across {best['case_count']} historical peer incidents.",
            f"Supporting historical cases: {', '.join(best['disruption_ids'][:5])}.",
        ],
        "supporting_disruption_ids": best.get("disruption_ids", []),
        "recovery_statistics": {
            "median_full_recovery_days": best["median_full_recovery_days"],
            "median_production_impact_pct": best["median_production_impact_pct"],
            "median_revenue_loss_usd": best["median_revenue_loss_usd"],
            "effectiveness_score": best["effectiveness_score"],
            "case_count": best["case_count"],
        },
        "alternative_trade_offs": alt_trade_offs,
        "confidence": round(min(1.0, best["case_count"] / total_cases), 3),
        "confidence_basis": f"Evidence share: {best['case_count']} of {total_cases} retrieved cases used this response; not a probability.",
        "human_review_required": total_cases < 5 or best["case_count"] < 2,
        "evidence_grounding": "Strictly restricted to valid historical response types and observed peer events",
    }

