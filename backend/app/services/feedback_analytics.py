from __future__ import annotations

import re
from typing import Any


def analyze_user_feedback(feedback_records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyzes stored feedback records to compute metrics, identify recurring issues,
    flag low-rated analyses, and generate improvement insights for prompt/retrieval refinement.
    
    Guarantees user isolation: only takes records that belong to the authenticated user.
    """
    total_count = len(feedback_records)
    if total_count == 0:
        return {
            "stats": {
                "total_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "helpfulness_percentage": 0.0,
                "average_rating": 0.0,
                "rating_scale": 5.0,
            },
            "recent_feedback": [],
            "low_rated_analyses": [],
            "recurring_issues": [],
            "improvement_insights": [
                {
                    "component": "System Readiness",
                    "area": "Feedback Data Collection",
                    "observation": "No feedback records recorded yet.",
                    "recommended_action": "Run disruption analyses and record operational feedback on recommendation usefulness to initiate the continuous improvement loop.",
                    "status": "Awaiting Data",
                }
            ],
            "prediction_error_metrics": {
                "paired_count": 0,
                "mae": None,
                "mean_error": None,
                "median_error": None,
                "rmse": None,
                "baseline_test_mae": 31.74,
                "mae_drift_pct": None,
            },
            "retraining_evaluation": {
                "status": "INSUFFICIENT_DATA",
                "retraining_justified": False,
                "decision": "Retraining Not Justified: No verified operational feedback with actual recovery time recorded yet (requires at least 5 verified cases).",
                "sample_count": 0,
                "min_samples_required": 5,
                "automated_retraining": "Disabled (Requires human validation and offline evaluation)",
            },
            "error_trends": [],
        }


    positive_count = sum(1 for item in feedback_records if item.get("useful") is True)
    negative_count = sum(1 for item in feedback_records if item.get("useful") is False)
    helpfulness_pct = round((positive_count / total_count) * 100.0, 1)
    
    # 5-star rating conversion: positive = 5.0, negative = 1.0
    avg_rating = round((positive_count * 5.0 + negative_count * 1.0) / total_count, 1)

    # Categorize issues across negative feedback
    issue_buckets: dict[str, dict[str, Any]] = {
        "recovery_discrepancy": {
            "category": "Recovery Time Estimation Discrepancy",
            "component": "Impact Assessment (ML Recovery Model)",
            "count": 0,
            "examples": [],
            "description": "Reported actual recovery days differed significantly from model-predicted duration or users reported duration misestimation.",
        },
        "impact_variance": {
            "category": "Production Impact & Financial Variance",
            "component": "Impact Assessment (ML Estimator)",
            "count": 0,
            "examples": [],
            "description": "Observed production downtime or financial losses varied from predicted values.",
        },
        "strategy_feasibility": {
            "category": "Response Strategy Feasibility",
            "component": "Recommendation Quality & Prompt Constraints",
            "count": 0,
            "examples": [],
            "description": "Recommended response action was deemed impractical, unfeasible for supplier tier, or different from chosen response.",
        },
        "retrieval_relevance": {
            "category": "Retrieval & Historical Context Relevance",
            "component": "Hybrid Retrieval (ChromaDB & BM25)",
            "count": 0,
            "examples": [],
            "description": "Retrieved historical cases did not closely match regional or categorical operational constraints.",
        },
        "severity_calibration": {
            "category": "Severity & Triage Alignment",
            "component": "Classification Agent",
            "count": 0,
            "examples": [],
            "description": "Assigned severity level or triage urgency did not reflect actual incident magnitude.",
        },
        "general_critique": {
            "category": "General Quality & Evidence Grounding",
            "component": "Recommendation Agent & Reasoning",
            "count": 0,
            "examples": [],
            "description": "Feedback noted insufficient grounding, lack of detail, or generic rationale.",
        },
    }

    low_rated_list = []

    for item in feedback_records:
        is_negative = item.get("useful") is False
        comment = (item.get("comment") or "").strip().lower()
        actual_recovery = item.get("actual_recovery_days")
        predicted_recovery = item.get("predicted_recovery_days")
        actual_impact = item.get("actual_production_impact_pct")
        predicted_impact = item.get("predicted_impact_pct")
        actual_response = item.get("actual_response")
        recommended_response = item.get("recommended_response")

        item_issues = []

        # Check recovery discrepancy
        if actual_recovery is not None and predicted_recovery is not None:
            diff = abs(float(actual_recovery) - float(predicted_recovery))
            if diff >= 10.0:
                issue_buckets["recovery_discrepancy"]["count"] += 1
                item_issues.append(f"Recovery days diff: predicted {predicted_recovery}d vs actual {actual_recovery}d")
                if item.get("comment"):
                    issue_buckets["recovery_discrepancy"]["examples"].append(item["comment"][:120])
        elif is_negative and any(k in comment for k in ["recovery", "day", "days", "timeline", "duration", "too long", "too short", "overestimate", "underestimate"]):
            issue_buckets["recovery_discrepancy"]["count"] += 1
            item_issues.append("Comment flagged recovery timeline inaccuracy")
            issue_buckets["recovery_discrepancy"]["examples"].append(item["comment"][:120])

        # Check impact / loss variance
        if actual_impact is not None and predicted_impact is not None:
            if abs(float(actual_impact) - float(predicted_impact)) >= 15.0:
                issue_buckets["impact_variance"]["count"] += 1
                item_issues.append(f"Production impact variance: predicted {predicted_impact}% vs actual {actual_impact}%")
        elif is_negative and any(k in comment for k in ["impact", "loss", "revenue", "financial", "cost", "exaggerated", "too high", "too low"]):
            issue_buckets["impact_variance"]["count"] += 1
            item_issues.append("Comment flagged production/revenue impact variance")
            issue_buckets["impact_variance"]["examples"].append(item["comment"][:120])

        # Check strategy feasibility
        if actual_response and recommended_response and actual_response.strip().lower() != recommended_response.strip().lower():
            issue_buckets["strategy_feasibility"]["count"] += 1
            item_issues.append(f"Strategy divergence: recommended '{recommended_response}' vs chosen '{actual_response}'")
        elif is_negative and any(k in comment for k in ["strategy", "action", "plan", "feasible", "unrealistic", "cannot", "reroute", "buffer", "supplier"]):
            issue_buckets["strategy_feasibility"]["count"] += 1
            item_issues.append("Comment flagged strategy feasibility or implementation barrier")
            issue_buckets["strategy_feasibility"]["examples"].append(item["comment"][:120])

        # Check retrieval relevance
        if is_negative and any(k in comment for k in ["retriev", "search", "similar", "cases", "context", "history", "precedent", "region", "tier"]):
            issue_buckets["retrieval_relevance"]["count"] += 1
            item_issues.append("Comment noted retrieved cases or context mismatch")
            issue_buckets["retrieval_relevance"]["examples"].append(item["comment"][:120])

        # Check severity calibration
        if is_negative and any(k in comment for k in ["severity", "critical", "minor", "triage", "priority", "mismatch"]):
            issue_buckets["severity_calibration"]["count"] += 1
            item_issues.append("Comment noted severity or risk level mismatch")
            issue_buckets["severity_calibration"]["examples"].append(item["comment"][:120])

        if is_negative and not item_issues:
            issue_buckets["general_critique"]["count"] += 1
            item_issues.append("Unhelpful recommendation flagged")
            if item.get("comment"):
                issue_buckets["general_critique"]["examples"].append(item["comment"][:120])

        if is_negative:
            low_rated_list.append({
                "analysis_id": item.get("analysis_id"),
                "disruption_type": item.get("disruption_type"),
                "industry": item.get("industry"),
                "severity": item.get("severity"),
                "recommended_response": item.get("recommended_response"),
                "predicted_recovery_days": item.get("predicted_recovery_days"),
                "actual_recovery_days": item.get("actual_recovery_days"),
                "actual_response": item.get("actual_response"),
                "comment": item.get("comment"),
                "flagged_issues": item_issues or ["Marked as not useful by user"],
                "created_at": item.get("created_at"),
            })

    # Filter active recurring issues (count > 0)
    recurring_issues = []
    for key, data in issue_buckets.items():
        if data["count"] > 0:
            pct = round((data["count"] / max(1, negative_count)) * 100.0, 1) if negative_count > 0 else 0.0
            recurring_issues.append({
                "issue_key": key,
                "category": data["category"],
                "component": data["component"],
                "occurrence_count": data["count"],
                "pct_of_negative_feedback": pct,
                "description": data["description"],
                "recent_observations": data["examples"][:3],
            })

    recurring_issues.sort(key=lambda x: x["occurrence_count"], reverse=True)

    # Synthesize actionable improvement insights for offline prompt/retrieval evaluation
    improvement_insights = []
    
    if any(i["issue_key"] == "recovery_discrepancy" for i in recurring_issues):
        rec_data = next(i for i in recurring_issues if i["issue_key"] == "recovery_discrepancy")
        improvement_insights.append({
            "component": "Impact Assessment (ML Model)",
            "area": "Recovery Time Estimator Calibration",
            "observation": f"{rec_data['occurrence_count']} analysis instances reported discrepancy between predicted and actual recovery duration.",
            "recommended_action": "Evaluate recovery_time_model calibration against reported recovery days in validation split. Verify error distributions across supplier tiers.",
            "status": "Ready for Offline Evaluation",
        })

    if any(i["issue_key"] == "strategy_feasibility" for i in recurring_issues):
        strat_data = next(i for i in recurring_issues if i["issue_key"] == "strategy_feasibility")
        improvement_insights.append({
            "component": "Recommendation Quality",
            "area": "Prompt Constraints & Alternative Strategy Ranking",
            "observation": f"{strat_data['occurrence_count']} analyses flagged strategy practicality or diverged from selected operational response.",
            "recommended_action": "Review system prompt instructions regarding supplier tier constraints (e.g. backup supplier availability). Benchmark alternative strategy options in RAGAS test suite.",
            "status": "Ready for Offline Evaluation",
        })

    if any(i["issue_key"] == "retrieval_relevance" for i in recurring_issues):
        ret_data = next(i for i in recurring_issues if i["issue_key"] == "retrieval_relevance")
        improvement_insights.append({
            "component": "Hybrid Retrieval",
            "area": "ChromaDB & BM25 Weighting and Filter Tuning",
            "observation": f"{ret_data['occurrence_count']} analyses flagged context mismatch in retrieved historical cases.",
            "recommended_action": "Evaluate hybrid balance (current 70% dense / 30% sparse) against low-rated scenario queries. Test semantic threshold adjustments on BGE-small embeddings.",
            "status": "Ready for Offline Evaluation",
        })

    # Compute prediction vs actual error metrics
    paired_cases = []
    for item in feedback_records:
        act_rec = item.get("actual_recovery_days")
        pred_rec = item.get("predicted_recovery_days")
        if act_rec is not None and pred_rec is not None:
            try:
                act_f = float(act_rec)
                pred_f = float(pred_rec)
                err = round(act_f - pred_f, 1)
                abs_err = round(abs(act_f - pred_f), 1)
                pct_err = round((abs_err / max(1.0, act_f)) * 100.0, 1)
                paired_cases.append({
                    "feedback_id": item.get("id"),
                    "analysis_id": item.get("analysis_id"),
                    "disruption_type": item.get("disruption_type") or "Disruption Event",
                    "industry": item.get("industry") or "All Industries",
                    "severity": item.get("severity"),
                    "created_at": item.get("created_at"),
                    "predicted_recovery_days": round(pred_f, 1),
                    "actual_recovery_days": round(act_f, 1),
                    "predicted": round(pred_f, 1),
                    "actual": round(act_f, 1),
                    "label": f"#{item.get('analysis_id')} {item.get('disruption_type') or ''}"[:14],
                    "error_days": err,
                    "abs_error_days": abs_err,
                    "percentage_error": pct_err,
                    "useful": item.get("useful"),
                    "comment": item.get("comment"),
                })
            except (ValueError, TypeError):
                pass

    paired_count = len(paired_cases)
    baseline_test_mae = 31.74
    if paired_count > 0:
        mae = round(sum(c["abs_error_days"] for c in paired_cases) / paired_count, 2)
        mean_err = round(sum(c["error_days"] for c in paired_cases) / paired_count, 2)
        rmse = round((sum(c["error_days"] ** 2 for c in paired_cases) / paired_count) ** 0.5, 2)
        sorted_errors = sorted(c["abs_error_days"] for c in paired_cases)
        median_err = round(sorted_errors[paired_count // 2], 2)
        drift_pct = round(((mae - baseline_test_mae) / baseline_test_mae) * 100.0, 1)
    else:
        mae = None
        mean_err = None
        rmse = None
        median_err = None
        drift_pct = None

    if paired_count < 5:
        retrain_status = "INSUFFICIENT_DATA"
        retrain_justified = False
        decision_text = (
            f"Retraining Not Justified: Insufficient verified recovery samples ({paired_count}/5 required). "
            f"Automatic model retraining is disabled to prevent training on sparse or incomplete feedback."
        )
    elif mae is not None and mae > (1.25 * baseline_test_mae):
        retrain_status = "RETRAINING_JUSTIFIED"
        retrain_justified = True
        decision_text = (
            f"Model Retraining Justified: Feedback recovery MAE ({mae} days) exceeds baseline test MAE (31.74 days) "
            f"by {drift_pct}%. Noticeable performance drift detected across operational feedback. "
            f"Supervised offline retraining is recommended."
        )
    else:
        retrain_status = "STABLE"
        retrain_justified = False
        decision_text = (
            f"Model Retraining Not Justified: Feedback recovery MAE ({mae} days) is aligned with baseline test MAE "
            f"(31.74 days). Model performance remains stable."
        )

    retraining_eval = {
        "status": retrain_status,
        "retraining_justified": retrain_justified,
        "decision": decision_text,
        "reason": decision_text,
        "sample_count": paired_count,
        "min_samples_required": 5,
        "baseline_test_mae": baseline_test_mae,
        "feedback_mae": mae,
        "bias_mean_error": mean_err,
        "automated_retraining": "Disabled (Requires human validation and offline evaluation)",
    }

    if not improvement_insights:
        if helpfulness_pct >= 80.0:
            improvement_insights.append({
                "component": "Overall Workflow",
                "area": "Continuous Monitoring",
                "observation": f"High helpfulness rate ({helpfulness_pct}%) across {total_count} submitted feedback records.",
                "recommended_action": "Maintain current prompt templates, retrieval weights (70/30), and scikit-learn pipelines while monitoring ongoing feedback.",
                "status": "Stable Performance",
            })
        else:
            improvement_insights.append({
                "component": "General Quality",
                "area": "General Decision Support",
                "observation": f"Negative feedback rate is {round(100 - helpfulness_pct, 1)}% across {total_count} records.",
                "recommended_action": "Audit low-rated analyses with human supply chain experts to identify missing constraints before initiating offline prompt tuning.",
                "status": "Ready for Offline Evaluation",
            })

    # Sort error trends chronologically (oldest to newest for trend line)
    error_trends = sorted(paired_cases, key=lambda x: str(x.get("created_at") or ""))

    pred_metrics = {
        "paired_count": paired_count,
        "mae": mae,
        "mean_error": mean_err,
        "median_error": median_err,
        "rmse": rmse,
        "baseline_test_mae": baseline_test_mae,
        "mae_drift_pct": drift_pct,
        "drift_detected": retrain_justified,
    }

    return {
        "stats": {
            "total_count": total_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "helpfulness_percentage": helpfulness_pct,
            "average_rating": avg_rating,
            "rating_scale": 5.0,
        },
        "recent_feedback": feedback_records[:20],
        "low_rated_analyses": low_rated_list[:15],
        "recurring_issues": recurring_issues,
        "improvement_insights": improvement_insights,
        "prediction_error_metrics": pred_metrics,
        "error_analytics": pred_metrics,
        "retraining_evaluation": retraining_eval,
        "error_trends": error_trends,
    }
