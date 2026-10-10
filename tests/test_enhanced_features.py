import pytest
from app.services.analysis import simulate_what_if_strategies, recommendation_from_history, STRATEGY_PROFILES
from app.services.feedback_analytics import analyze_user_feedback
from app.services.supplier_risk import get_supplier_risk_overview
from app.ml.predictor import predictions


@pytest.fixture
def mock_retrieved_cases():
    return [
        {
            "disruption_id": "DISR-00101",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_tier": "2",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Medium",
            "has_backup_supplier": True,
            "disruption_severity": 4,
            "production_impact_pct": 24.5,
            "revenue_loss_usd": 1500000.0,
            "response_type": "Alternative Supplier",
            "response_time_days": 12.0,
            "partial_recovery_days": 28.0,
            "full_recovery_days": 42.0,
            "permanent_supplier_change": False,
        },
        {
            "disruption_id": "DISR-00102",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_tier": "2",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Large",
            "has_backup_supplier": True,
            "disruption_severity": 4,
            "production_impact_pct": 28.0,
            "revenue_loss_usd": 1800000.0,
            "response_type": "Alternative Supplier",
            "response_time_days": 14.0,
            "partial_recovery_days": 30.0,
            "full_recovery_days": 48.0,
            "permanent_supplier_change": False,
        },
        {
            "disruption_id": "DISR-00201",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_tier": "2",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Medium",
            "has_backup_supplier": False,
            "disruption_severity": 4,
            "production_impact_pct": 45.0,
            "revenue_loss_usd": 3800000.0,
            "response_type": "Inventory Buffer",
            "response_time_days": 8.0,
            "partial_recovery_days": 35.0,
            "full_recovery_days": 75.0,
            "permanent_supplier_change": False,
        },
        {
            "disruption_id": "DISR-00301",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_tier": "2",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Small",
            "has_backup_supplier": False,
            "disruption_severity": 4,
            "production_impact_pct": 52.0,
            "revenue_loss_usd": 5200000.0,
            "response_type": "Production Reroute",
            "response_time_days": 18.0,
            "partial_recovery_days": 45.0,
            "full_recovery_days": 82.0,
            "permanent_supplier_change": True,
        },
    ]


# -------------------------------------------------------------------------
# 1. What-if Disruption Simulator Tests
# -------------------------------------------------------------------------
def test_what_if_simulator_generation(mock_retrieved_cases):
    results = simulate_what_if_strategies(mock_retrieved_cases)
    assert len(results) >= 4

    strategy_names = [r["response_type"] for r in results]
    assert "Alternative Supplier" in strategy_names
    assert "Inventory Buffer" in strategy_names
    assert "Production Reroute" in strategy_names

    # Check that required fields are present
    for r in results:
        assert "historical_median_full_recovery_days" in r
        assert "historical_median_production_impact_pct" in r
        assert "historical_median_revenue_loss_usd" in r
        assert "effectiveness_score" in r
        assert "advantages" in r
        assert "disadvantages" in r
        assert "trade_offs" in r
        assert "distinction_note" in r
        assert "Historical statistics reflect empirical outcomes" in r["distinction_note"]

    # Check recommended strategy
    recommended = [r for r in results if r["is_recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["response_type"] == "Alternative Supplier"
    assert "Selected as primary strategy" in recommended[0]["recommendation_reason"]
    assert "DISR-00101" in recommended[0]["supporting_disruption_ids"]


# -------------------------------------------------------------------------
# 2. Evidence-Based Recommendation Explanations Tests
# -------------------------------------------------------------------------
def test_evidence_based_recommendation_explanations(mock_retrieved_cases):
    from app.services.analysis import response_analysis
    groups = response_analysis(mock_retrieved_cases)
    rec = recommendation_from_history(groups, total_cases=len(mock_retrieved_cases))

    assert rec["recommended_response"] in ["Alternative Supplier", "Inventory Buffer", "Production Reroute"]
    assert "supporting_disruption_ids" in rec
    assert len(rec["supporting_disruption_ids"]) >= 1
    assert "recovery_statistics" in rec
    assert rec["recovery_statistics"]["median_full_recovery_days"] is not None
    assert rec["recovery_statistics"]["case_count"] >= 1

    # Alternative trade offs
    assert "alternative_trade_offs" in rec
    assert len(rec["alternative_trade_offs"]) >= 1
    for alt in rec["alternative_trade_offs"]:
        assert "trade_off" in alt
        assert "response_type" in alt


# -------------------------------------------------------------------------
# 3. Feedback-Based Model Monitoring & Retraining Evaluation Tests
# -------------------------------------------------------------------------
def test_feedback_monitoring_under_5_samples():
    records = [
        {
            "id": 1,
            "analysis_id": 101,
            "useful": True,
            "actual_recovery_days": 40.0,
            "predicted_recovery_days": 35.0,
            "comment": "Accurate prediction",
        },
        {
            "id": 2,
            "analysis_id": 102,
            "useful": False,
            "actual_recovery_days": 85.0,
            "predicted_recovery_days": 42.0,
            "comment": "Recovered much slower than expected",
        },
    ]
    result = analyze_user_feedback(records)
    metrics = result["prediction_error_metrics"]
    assert metrics["paired_count"] == 2
    assert metrics["mae"] == pytest.approx(24.0, 0.1)

    eval_result = result["retraining_evaluation"]
    assert eval_result["status"] == "INSUFFICIENT_DATA"
    assert eval_result["retraining_justified"] is False
    assert "5 required" in eval_result["decision"]
    assert "Disabled" in eval_result["automated_retraining"]


def test_feedback_monitoring_retraining_justified_on_drift():
    # 5 paired samples with severe error (MAE > 1.25 * 31.74 = 39.675)
    records = [
        {"id": i, "analysis_id": 100 + i, "useful": False, "actual_recovery_days": 120.0, "predicted_recovery_days": 40.0}
        for i in range(1, 6)
    ]
    result = analyze_user_feedback(records)
    metrics = result["prediction_error_metrics"]
    assert metrics["paired_count"] == 5
    assert metrics["mae"] == 80.0

    eval_result = result["retraining_evaluation"]
    assert eval_result["status"] == "RETRAINING_JUSTIFIED"
    assert eval_result["retraining_justified"] is True
    assert "Model Retraining Justified" in eval_result["decision"]


# -------------------------------------------------------------------------
# 4. Prediction Reliability and Uncertainty Tests
# -------------------------------------------------------------------------
def test_prediction_uncertainty_and_reliability():
    input_data = {
        "disruption_type": "Port Congestion",
        "industry": "Electronics",
        "supplier_tier": "2",
        "supplier_region": "Asia-Pacific",
        "supplier_size": "Medium",
        "has_backup_supplier": True,
        "disruption_severity": 4,
    }
    result = predictions(input_data)
    assert result["available"] is True

    preds = result["predictions"]
    # Check production impact uncertainty
    assert "uncertainty" in preds["production_impact"]
    assert preds["production_impact"]["uncertainty"]["test_mae"] == 7.99
    assert preds["production_impact"]["uncertainty"]["interval_low"] <= preds["production_impact"]["prediction"]

    # Check recovery time uncertainty
    assert "uncertainty" in preds["recovery_time"]
    assert preds["recovery_time"]["uncertainty"]["test_mae"] == 31.7
    assert preds["recovery_time"]["uncertainty"]["test_r2"] == pytest.approx(0.3689, 0.01)

    # Check reliability assessment
    assert "reliability" in result
    rel = result["reliability"]
    assert "reliability_score" in rel
    assert "reliability_level" in rel
    assert "reasons" in rel
    assert len(rel["reasons"]) >= 3
    assert "distinction_notice" in rel


# -------------------------------------------------------------------------
# 5. Supplier Risk Monitoring Dashboard Tests
# -------------------------------------------------------------------------
def test_supplier_risk_overview():
    overview = get_supplier_risk_overview(db=None, limit=20)
    assert overview["total_records_analyzed"] > 0
    assert "HISTORICAL OPERATIONAL VULNERABILITY" in overview["notice"]
    assert len(overview["segments"]) > 0

    first_segment = overview["segments"][0]
    assert "composite_risk_score" in first_segment
    assert "risk_category" in first_segment
    assert "vulnerability_reasons" in first_segment
    assert "supporting_disruption_ids" in first_segment
    assert first_segment["composite_risk_score"] >= 0.0
    assert first_segment["composite_risk_score"] <= 100.0

    # Test filtering by region
    filtered = get_supplier_risk_overview(db=None, region="Asia-Pacific", limit=10)
    for seg in filtered["segments"]:
        assert seg["supplier_region"].lower() == "asia-pacific"
