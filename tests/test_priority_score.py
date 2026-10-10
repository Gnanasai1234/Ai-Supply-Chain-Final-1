import logging
import pytest
from app.services.analysis import load_priority_reference, priority_details
from app.agents.workflow import analyze_priority


def test_user_disruption_priority_issue_case():
    """
    Test the exact case reported by the user:
    - Severity 5/5
    - Predicted production impact 64.9%
    - Predicted financial loss $11,549,566
    Expected:
    - Empirical score is ~97.2 / 100
    - Priority label is CRITICAL (since score >= 90)
    - Non-zero percentiles for all components
    """
    reference = load_priority_reference(None)
    assert reference and len(reference["disruption_severity"]) > 0

    result = priority_details(
        severity=5,
        production_impact_pct=64.9,
        revenue_loss_usd=11549566.0,
        reference=reference,
        impact_source="Predicted production impact",
        loss_source="Predicted revenue loss",
    )

    assert result["priority_score"] == 97.2
    assert result["priority_level"] == "CRITICAL"
    reasons = {r["factor"]: r for r in result["priority_reasons"]}
    assert reasons["Disruption severity"]["dataset_percentile"] == 100.0
    assert reasons["Predicted production impact"]["dataset_percentile"] > 90.0
    assert reasons["Predicted revenue loss"]["dataset_percentile"] > 90.0


def test_low_severity_case():
    """
    Test a minor routine disruption:
    - Severity 1/5
    - Production impact 5%
    - Revenue loss $25,000
    Expected:
    - Priority score < 40
    - Priority label is LOW
    """
    result = priority_details(
        severity=1,
        production_impact_pct=5.0,
        revenue_loss_usd=25000.0,
        reference=None,  # Tests automatic loading of reference
        impact_source="Observed production impact",
        loss_source="Observed revenue loss (USD)",
    )

    assert 0.0 <= result["priority_score"] < 40.0
    assert result["priority_level"] == "LOW"


def test_medium_and_high_severity_thresholds():
    """
    Test that medium and high thresholds match specifications:
    - MEDIUM: 40 <= score < 70
    - HIGH: 70 <= score < 90
    - CRITICAL: score >= 90
    """
    ref = load_priority_reference(None)

    # Moderate case
    medium_res = priority_details(3, 30.0, 1500000.0, ref)
    assert 40.0 <= medium_res["priority_score"] < 70.0
    assert medium_res["priority_level"] == "MEDIUM"

    # High severity case (expected in 70 - 89 range -> HIGH)
    high_res = priority_details(4, 40.0, 3000000.0, ref)
    assert 70.0 <= high_res["priority_score"] < 90.0
    assert high_res["priority_level"] == "HIGH"


def test_workflow_priority_with_missing_observed_values():
    """
    Test workflow agent analyze_priority:
    When observed production_impact_pct and revenue_loss_usd are None in input,
    it must properly use predicted values from state['predictions'].
    """
    state = {
        "validated_input": {
            "disruption_severity": 5,
            "production_impact_pct": None,
            "revenue_loss_usd": None,
        },
        "predictions": {
            "predictions": {
                "production_impact": {"prediction": 64.9},
                "financial_loss": {"prediction": 11549566.0},
            }
        },
        "priority_reference": {},  # empty dict simulates empty MySQL table before fix
        "db": None,
    }

    output = analyze_priority(state)
    priority = output["priority"]

    assert priority["available"] is True
    assert priority["priority_score"] == 97.2
    assert priority["priority_level"] == "CRITICAL"
    factors = [r["factor"] for r in priority["priority_reasons"]]
    assert "Predicted production impact" in factors
    assert "Predicted revenue loss (USD)" in factors


def test_priority_score_bounds_and_numeric():
    """
    Test that extreme inputs (e.g. negative or exorbitant numbers)
    produce scores strictly bounded between 0.0 and 100.0.
    """
    ref = load_priority_reference(None)

    # Extreme low
    low = priority_details(0, -50.0, -1000.0, ref)
    assert low["priority_score"] >= 0.0
    assert isinstance(low["priority_score"], float)

    # Extreme high
    high = priority_details(10, 999.0, 1e12, ref)
    assert high["priority_score"] <= 100.0
    assert isinstance(high["priority_score"], float)
    assert high["priority_level"] == "CRITICAL"


def test_priority_logging(caplog):
    """
    Test that clear logging is emitted with input values and final priority score.
    """
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        priority_details(
            severity=5,
            production_impact_pct=64.9,
            revenue_loss_usd=11549566.0,
            reference=None,
        )

    assert any(
        "Priority calculated: score=97.2, level=CRITICAL" in record.message
        for record in caplog.records
    )
