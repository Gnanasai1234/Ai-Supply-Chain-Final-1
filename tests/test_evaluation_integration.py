import json
from starlette.testclient import TestClient

from app.database import Base, engine
from app.evaluation_service import (
    calculate_overall_evaluation,
    evaluate_ragas_metrics,
    log_evaluation_console,
)
from app.main import app


def test_ragas_metrics_dynamic_calculation():
    """Test that RAGAS correctly evaluates the actual generated recommendation and retrieved context."""
    input_data = {
        "disruption_type": "Logistics Delay",
        "industry": "Automotive",
        "supplier_region": "Asia",
        "supplier_tier": "1",
        "disruption_severity": 4,
        "description": "Port congestion causing severe delay in Asia automotive parts",
    }
    recommendation = {
        "summary": "Activate dual sourcing and expedite container shipping from Asia ports.",
        "recommended_response": "Dual Sourcing",
        "recommended_actions": [
            "Contract emergency carrier capacity in Asia",
            "Reroute critical automotive components via air freight",
        ],
    }
    retrieved_cases = [
        {
            "disruption_id": "DIS-0010",
            "disruption_type": "Logistics Delay",
            "industry": "Automotive",
            "supplier_region": "Asia",
            "supplier_tier": "1",
            "disruption_severity": 4,
            "response_type": "Dual Sourcing",
            "full_recovery_days": 25,
            "similarity_score": 0.92,
        },
        {
            "disruption_id": "DIS-0015",
            "disruption_type": "Port Congestion",
            "industry": "Automotive",
            "supplier_region": "Asia",
            "supplier_tier": "1",
            "disruption_severity": 3,
            "response_type": "Buffer Stock",
            "full_recovery_days": 20,
            "similarity_score": 0.81,
        },
    ]

    ragas = evaluate_ragas_metrics(input_data, recommendation, retrieved_cases, threshold=0.70)
    assert ragas["status"] == "available"
    assert ragas["framework"] == "RAGAS"
    assert "overall_pct" in ragas
    assert 0.0 <= ragas["overall_score"] <= 1.0

    # 4 core RAGAS metrics
    metrics = ragas["metrics"]
    assert "faithfulness" in metrics
    assert "answer_relevancy" in metrics
    assert "context_precision" in metrics
    assert "context_recall" in metrics

    # Real evaluated scores
    assert 0.0 <= metrics["faithfulness"]["score"] <= 1.0
    assert metrics["faithfulness"]["score"] > 0.0
    assert metrics["faithfulness"]["reason"]

    # Context precision should reflect top-ranked relevant cases
    assert metrics["context_precision"]["score"] > 0.0

    # Context recall should reflect coverage of query facets (type, industry, region, sev, tier)
    assert metrics["context_recall"]["score"] >= 0.60


def test_ragas_metrics_empty_retrieval():
    """Test that RAGAS produces 0.0 scores and warnings when no context is retrieved (no fake scores)."""
    input_data = {"disruption_type": "Cyber Attack", "disruption_severity": 5}
    recommendation = {
        "summary": "Implement network isolation",
        "recommended_response": "Cyber Isolation",
        "recommended_actions": ["Disconnect servers"],
    }
    ragas = evaluate_ragas_metrics(input_data, recommendation, [], threshold=0.70)
    assert ragas["status"] == "available"
    # With no retrieved cases, faithfulness, precision, and recall are 0.0
    assert ragas["metrics"]["faithfulness"]["score"] == 0.0
    assert ragas["metrics"]["context_precision"]["score"] == 0.0
    assert ragas["metrics"]["context_recall"]["score"] == 0.0
    assert ragas["passed"] is False


def test_calculate_overall_evaluation_ragas_only():
    """Test that calculate_overall_evaluation uses RAGAS as the ONLY evaluation method."""
    input_data = {
        "disruption_type": "Factory Incident",
        "industry": "Manufacturing",
        "supplier_region": "Europe",
        "disruption_severity": 4,
        "description": "Plant explosion halting assembly",
    }
    recommendation = {
        "summary": "Activate emergency backup supplier and escalate mitigation",
        "recommended_response": "Emergency Supplier Activation",
        "recommended_actions": [
            "Escalate immediately to operations committee",
            "Engage secondary supplier to prevent production halt",
        ],
    }
    retrieved_cases = [
        {
            "disruption_id": "DIS-0033",
            "disruption_type": "Factory Incident",
            "industry": "Manufacturing",
            "supplier_region": "Europe",
            "response_type": "Emergency Supplier Activation",
            "disruption_severity": 4,
            "similarity_score": 0.88,
        }
    ]

    report = calculate_overall_evaluation(input_data, recommendation, retrieved_cases, threshold=75.0)

    # RAGAS is the only evaluation method
    assert report["evaluation_method"] == "RAGAS"
    assert "ragas" in report
    assert "retrieval" not in report
    assert "deepeval" not in report
    assert "test_cases" not in report

    # Overall score equals RAGAS overall score
    assert report["overall_score_pct"] == report["ragas"]["overall_pct"]
    assert report["status"] in {"EXCELLENT", "GOOD"}


def test_overall_evaluation_warning_trigger():
    """Case with empty retrieval should trigger warnings and fail target threshold."""
    input_data = {"disruption_type": "Unknown Event", "disruption_severity": 5}
    recommendation = {"summary": "", "recommended_response": "", "recommended_actions": []}
    retrieved_cases = []

    eval_report = calculate_overall_evaluation(input_data, recommendation, retrieved_cases, threshold=75.0)
    assert eval_report["has_warning"] is True
    assert len(eval_report["warnings"]) > 0
    assert eval_report["overall_score_pct"] < 75.0
    assert eval_report["status"] in {"NEEDS IMPROVEMENT", "POOR"}


def test_api_evaluation_endpoints():
    """Test API /api/evaluation endpoint returns RAGAS evaluation."""
    Base.metadata.create_all(engine)
    client = TestClient(app)

    resp = client.get("/api/evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "available"
    assert data["evaluation_method"] == "RAGAS"
    assert "evaluations" in data
