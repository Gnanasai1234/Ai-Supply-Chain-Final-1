from __future__ import annotations

import asyncio
import httpx
import pytest
from app.models import AnalysisHistory, AnalysisRequest, Disruption, Feedback


def request(api_app, method: str, path: str, headers: dict | None = None, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, headers=headers, **kwargs)

    return asyncio.run(send())


def register_and_login(api_app, email: str = "test_feedback_user@example.com"):
    reg_resp = request(
        api_app,
        "POST",
        "/api/auth/register",
        json={"name": "Feedback Tester", "email": email, "password": "secure-password-123"},
    )
    token = reg_resp.json()["access_token"]
    user_id = reg_resp.json()["user"]["id"]
    return token, user_id


def test_feedback_analytics_and_user_isolation(api_app, session_factory):
    token_a, user_a_id = register_and_login(api_app, "user_a@example.com")
    headers_a = {"Authorization": f"Bearer {token_a}"}

    token_b, user_b_id = register_and_login(api_app, "user_b@example.com")
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # Seed analysis requests and analysis history for User A
    with session_factory.begin() as db:
        req_1 = AnalysisRequest(
            request_data={"disruption_type": "Port Congestion", "industry": "Electronics", "severity": 4},
            result_data={"status": "success"},
        )
        db.add(req_1)
        db.flush()
        req_1_id = req_1.id

        hist_1 = AnalysisHistory(
            user_id=user_a_id,
            analysis_request_id=req_1_id,
            query_text="Port Congestion Electronics Tier 2",
            search_text="port congestion electronics tier 2",
            request_data={"disruption_type": "Port Congestion", "industry": "Electronics", "disruption_severity": 4},
            response_data={
                "predictions": {
                    "predictions": {
                        "recovery_time": {"prediction": 60.0, "unit": "days"},
                        "production_impact": {"prediction": 45.0, "unit": "%"},
                        "financial_loss": {"prediction": 500000.0, "unit": "USD"},
                    }
                },
                "recommendations": {
                    "recommended_response": "Alternative Supplier",
                    "reasoning": ["Supplier switch recommended due to 60-day estimated delay"],
                },
            },
            disruption_type="Port Congestion",
            industry="Electronics",
            supplier_region="Asia-Pacific",
            supplier_tier="2",
            severity=4,
            final_provider="Google Gemini",
            final_model="gemini-3.5-flash-lite",
            fallback_used=True,
            provider_attempts=[],
            response_time_ms=850,
            status="success",
        )
        db.add(hist_1)

        req_2 = AnalysisRequest(
            request_data={"disruption_type": "Factory Incident", "industry": "Automotive", "severity": 5},
            result_data={"status": "success"},
        )
        db.add(req_2)
        db.flush()
        req_2_id = req_2.id

        hist_2 = AnalysisHistory(
            user_id=user_a_id,
            analysis_request_id=req_2_id,
            query_text="Factory Incident Automotive Tier 1",
            search_text="factory incident automotive tier 1",
            request_data={"disruption_type": "Factory Incident", "industry": "Automotive", "disruption_severity": 5},
            response_data={
                "predictions": {
                    "predictions": {
                        "recovery_time": {"prediction": 95.0, "unit": "days"},
                        "production_impact": {"prediction": 65.0, "unit": "%"},
                    }
                },
                "recommendations": {
                    "recommended_response": "Production Reroute",
                },
            },
            disruption_type="Factory Incident",
            industry="Automotive",
            supplier_region="Europe",
            supplier_tier="1",
            severity=5,
            final_provider="Anthropic Claude",
            final_model="claude-sonnet-4-5",
            fallback_used=False,
            provider_attempts=[],
            response_time_ms=1200,
            status="success",
        )
        db.add(hist_2)

    # 1. User A submits positive feedback for Analysis 1
    resp_sub_1 = request(
        api_app,
        "POST",
        "/api/feedback",
        headers=headers_a,
        json={
            "analysis_id": req_1_id,
            "useful": True,
            "comment": "Accurate assessment, alternative supplier worked well",
            "actual_recovery_days": 58.0,
            "actual_response": "Alternative Supplier",
        },
    )
    assert resp_sub_1.status_code == 201
    assert "id" in resp_sub_1.json()

    # 2. User A submits negative feedback for Analysis 2 with recovery discrepancy
    resp_sub_2 = request(
        api_app,
        "POST",
        "/api/feedback",
        headers=headers_a,
        json={
            "analysis_id": req_2_id,
            "useful": False,
            "comment": "Recovery time was overestimated; production reroute was not feasible due to tooling constraints",
            "actual_recovery_days": 40.0,  # 95 predicted vs 40 actual -> 55-day difference
            "actual_response": "Inventory Buffer",
        },
    )
    assert resp_sub_2.status_code == 201

    # 3. Query GET /api/feedback for User A
    resp_get_a = request(api_app, "GET", "/api/feedback", headers=headers_a)
    assert resp_get_a.status_code == 200
    data_a = resp_get_a.json()

    # Check stats
    stats_a = data_a["stats"]
    assert stats_a["total_count"] == 2
    assert stats_a["positive_count"] == 1
    assert stats_a["negative_count"] == 1
    assert stats_a["helpfulness_percentage"] == 50.0
    assert stats_a["average_rating"] == 3.0  # (5*1 + 1*1) / 2 = 3.0

    # Check low-rated analyses
    assert len(data_a["low_rated_analyses"]) == 1
    low = data_a["low_rated_analyses"][0]
    assert low["analysis_id"] == req_2_id
    assert low["disruption_type"] == "Factory Incident"
    assert low["predicted_recovery_days"] == 95.0
    assert low["actual_recovery_days"] == 40.0

    # Check recurring issues detection
    recurring = data_a["recurring_issues"]
    assert len(recurring) > 0
    issue_keys = [item["issue_key"] for item in recurring]
    assert "recovery_discrepancy" in issue_keys or "strategy_feasibility" in issue_keys

    # Check improvement insights
    insights = data_a["improvement_insights"]
    assert len(insights) > 0
    assert any("offline evaluation" in ins["recommended_action"].lower() or "calibration" in ins["recommended_action"].lower() for ins in insights)

    # 4. Check GET /api/feedback/stats endpoint
    resp_stats_a = request(api_app, "GET", "/api/feedback/stats", headers=headers_a)
    assert resp_stats_a.status_code == 200
    assert resp_stats_a.json()["stats"]["total_count"] == 2

    # 5. USER ISOLATION VERIFICATION:
    # Query GET /api/feedback for User B -> must be completely empty
    resp_get_b = request(api_app, "GET", "/api/feedback", headers=headers_b)
    assert resp_get_b.status_code == 200
    data_b = resp_get_b.json()
    assert data_b["stats"]["total_count"] == 0
    assert data_b["stats"]["positive_count"] == 0
    assert data_b["recent_feedback"] == []
    assert data_b["low_rated_analyses"] == []

    # User B attempts to submit feedback on User A's analysis request -> must be rejected with 404
    resp_cross = request(
        api_app,
        "POST",
        "/api/feedback",
        headers=headers_b,
        json={"analysis_id": req_1_id, "useful": True},
    )
    assert resp_cross.status_code == 404
