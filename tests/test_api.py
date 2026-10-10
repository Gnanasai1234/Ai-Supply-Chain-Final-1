from __future__ import annotations

import asyncio
import json
import logging
import warnings

import httpx
import pytest
from sklearn.exceptions import InconsistentVersionWarning
from sqlalchemy.exc import SQLAlchemyError

warnings.simplefilter("error", InconsistentVersionWarning)

from app import main as main_module
from app import llm as llm_module
from app.agents.workflow import build_workflow
from app.config import settings
from app.ml.predictor import initialize_models
from app.models import AnalysisHistory, AnalysisRequest, AuthSession, Recommendation, User
from app.security import hash_password, verify_password


def request(api_app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=api_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def register_user(api_app, email="user@example.com"):
    response = request(api_app, "POST", "/api/auth/register", json={"name": "Test User", "email": email, "password": "correct-horse-battery"})
    assert response.status_code == 201
    return response.json()


def test_health_endpoint_reports_runtime_status(api_app):
    response = request(api_app, "GET", "/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "unavailable"}
    assert {"database", "chroma", "bm25", "models", "artifact_errors"} <= body.keys()


def test_disruptions_list_returns_seeded_record(api_app):
    response = request(api_app, "GET", "/api/disruptions?industry=Electronics&limit=1")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["disruption_id"] == "DIS-TEST-0001"


def test_disruptions_reject_invalid_limit(api_app):
    response = request(api_app, "GET", "/api/disruptions?limit=0")
    assert response.status_code == 422


def test_analysis_rejects_empty_description(api_app):
    account = register_user(api_app)
    response = request(api_app, "POST", "/api/disruptions/analyze", json={"description": ""}, headers={"Authorization": f"Bearer {account['access_token']}"})
    assert response.status_code == 422


def test_analysis_requires_authenticated_user(api_app):
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    assert request(api_app, "POST", "/api/disruptions/analyze", json=payload).status_code == 401


def test_auth_registration_login_logout_and_password_hash(api_app, session_factory):
    account = register_user(api_app, "alice@example.com")
    assert "password_hash" not in account["user"]
    with session_factory() as db:
        user = db.query(User).filter_by(email="alice@example.com").one()
        assert user.password_hash != "correct-horse-battery"
        assert user.password_hash.startswith("scrypt$")
        assert verify_password("correct-horse-battery", user.password_hash)
        assert not verify_password("wrong-password", user.password_hash)
        auth_session = db.query(AuthSession).one()
        assert auth_session.token_hash != account["access_token"]
        assert len(auth_session.token_hash) == 64
    duplicate = request(api_app, "POST", "/api/auth/register", json={"name": "Alice", "email": "ALICE@example.com", "password": "different-password"})
    assert duplicate.status_code == 409
    wrong_password = request(api_app, "POST", "/api/auth/login", json={"email": "alice@example.com", "password": "incorrect-password"})
    assert wrong_password.status_code == 401
    login = request(api_app, "POST", "/api/auth/login", json={"email": "ALICE@example.com", "password": "correct-horse-battery"})
    assert login.status_code == 200
    token = login.json()["access_token"]
    me = request(api_app, "GET", "/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"
    logout = request(api_app, "POST", "/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout.status_code == 200
    assert request(api_app, "GET", "/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_history_requires_auth_and_isolated_filters_search_pagination_delete(api_app, session_factory):
    assert request(api_app, "GET", "/api/history").status_code == 401
    alice = register_user(api_app, "alice@example.com")
    bob = register_user(api_app, "bob@example.com")
    from datetime import datetime, timedelta

    def seed_history(user_id, *, description, provider, model, severity, industry, days_ago=0):
        response_body = {"recommendation": {"summary": f"Evidence summary for {description}", "recommended_response": "Alternative Supplier"}}
        with session_factory.begin() as db:
            row = AnalysisHistory(
                user_id=user_id, query_text=description, search_text=f"{description} {provider} {industry}",
                request_data={"description": description}, response_data=response_body,
                disruption_type="Port Congestion", industry=industry, supplier_region="Asia-Pacific",
                supplier_tier="2", severity=severity, final_provider=provider, final_model=model,
                fallback_used=provider == "Deterministic Fallback", provider_attempts=[], response_time_ms=1200,
                status="success", created_at=datetime.now() - timedelta(days=days_ago),
            )
            db.add(row)
            db.flush()
            return row.id

    alice_first = seed_history(alice["user"]["id"], description="Electronics port event alpha", provider="Anthropic Claude", model="claude-sonnet-4-5", severity=4, industry="Electronics")
    seed_history(alice["user"]["id"], description="Automotive port event beta", provider="Deterministic Fallback", model="extractive", severity=3, industry="Automotive", days_ago=9)
    bob_history = seed_history(bob["user"]["id"], description="Bob only supplier event", provider="Google Gemini", model="gemini-3.5-flash-lite", severity=5, industry="Electronics")
    alice_headers = {"Authorization": f"Bearer {alice['access_token']}"}
    bob_headers = {"Authorization": f"Bearer {bob['access_token']}"}
    today = datetime.now().date().isoformat()

    filtered = request(api_app, "GET", "/api/history", params={
        "date_preset": "custom", "date_from": today, "date_to": today, "provider": "Anthropic Claude", "severity": 4,
        "industry": "Electronics", "page": 1, "page_size": 1,
    }, headers=alice_headers)
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["id"] == alice_first
    searched = request(api_app, "GET", "/api/history", params={"search": "alpha"}, headers=alice_headers)
    assert searched.json()["total"] == 1
    page_one = request(api_app, "GET", "/api/history", params={"page": 1, "page_size": 1}, headers=alice_headers).json()
    page_two = request(api_app, "GET", "/api/history", params={"page": 2, "page_size": 1}, headers=alice_headers).json()
    assert (page_one["total"], page_one["total_pages"], len(page_one["items"])) == (2, 2, 1)
    assert len(page_two["items"]) == 1
    last_week = request(api_app, "GET", "/api/history", params={"date_preset": "last_7_days"}, headers=alice_headers).json()
    assert last_week["total"] == 1
    stats = request(api_app, "GET", "/api/history/stats", headers=alice_headers).json()
    assert stats["total_analyses"] == 2
    assert stats["provider_counts"] == {"Deterministic Fallback": 1, "Anthropic Claude": 1}
    facets = request(api_app, "GET", "/api/history/filters", headers=alice_headers).json()
    assert set(facets["providers"]) == {"Anthropic Claude", "Deterministic Fallback"}
    assert "Automotive" in facets["industries"] or facets["industries"] == ["Electronics"]

    own_list = request(api_app, "GET", "/api/history", headers=alice_headers).json()
    assert own_list["total"] == 2
    assert all("Bob only" not in item["query"] for item in own_list["items"])
    bob_detail = request(api_app, "GET", f"/api/history/{bob_history}", headers=alice_headers)
    assert bob_detail.status_code == 404
    denied_delete = request(api_app, "DELETE", f"/api/history/{bob_history}", headers=alice_headers)
    assert denied_delete.status_code == 404
    assert request(api_app, "GET", f"/api/history/{bob_history}", headers=bob_headers).status_code == 200

    assert request(api_app, "DELETE", f"/api/history/{alice_first}", headers=alice_headers).json() == {"deleted": 1}
    assert request(api_app, "DELETE", "/api/history", headers=alice_headers).json() == {"deleted": 1}
    assert request(api_app, "GET", "/api/history", headers=alice_headers).json()["total"] == 0
    assert request(api_app, "GET", "/api/history", headers=bob_headers).json()["total"] == 1


def test_analysis_valid_shape_returns_explicit_unavailable_workflow(api_app, monkeypatch):
    account = register_user(api_app)
    monkeypatch.setattr(main_module, "workflow", None)
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/disruptions/analyze", json=payload, headers={"Authorization": f"Bearer {account['access_token']}"})
    assert response.status_code == 503
    assert "workflow unavailable" in response.json()["detail"].lower()


def test_unexpected_analysis_failure_is_logged_as_server_error(api_app, monkeypatch, caplog):
    account = register_user(api_app)
    class ReadyRetriever:
        pass

    class BrokenWorkflow:
        def invoke(self, state):
            raise RuntimeError("internal test failure")

    monkeypatch.setattr(main_module, "workflow", BrokenWorkflow())
    monkeypatch.setattr(main_module, "retriever", ReadyRetriever())
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/disruptions/analyze", json=payload,
                       headers={"Origin": "http://localhost:5173", "Authorization": f"Bearer {account['access_token']}"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis failed because of an unexpected server error. Check backend logs for details."}
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "Unexpected failure while running disruption analysis" in caplog.text
    assert "internal test failure" in caplog.text


def test_analysis_uses_anthropic_primary_and_returns_200(api_app, monkeypatch, caplog, capsys):
    caplog.set_level(logging.INFO)
    account = register_user(api_app)
    calls = []
    response_body = {
        "summary": "Retrieved cases support an alternative supplier response. " + ("More context. " * 55),
        "recommended_response": "Alternative Supplier", "alternative_responses": [],
        "reasoning": "This response is represented in the retrieved historical cases.",
        "evidence": ["DIS-HISTORY-1", "DIS-HISTORY-2"], "confidence": 0.5, "human_review_required": True,
    }

    def provider_response(request):
        calls.append(request)
        assert "api.anthropic.com" in str(request.url)
        assert "/v1/messages" in str(request.url)
        assert request.headers.get("x-api-key") == "test-anthropic-key"
        assert request.headers.get("anthropic-version") == "2023-06-01"
        assert json.loads(request.content)["model"] == "claude-sonnet-4-5"
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": json.dumps(response_body)}],
        })

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_model", "claude-sonnet-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")

    class Retriever:
        def search(self, db, query, filters, top_k):
            return [{
                "disruption_id": f"DIS-HISTORY-{index}", "response_type": "Alternative Supplier",
                "production_impact_pct": 40.0 + index, "revenue_loss_usd": 1000.0 + index,
                "response_time_days": 4, "partial_recovery_days": 20, "full_recovery_days": 60,
                "permanent_supplier_change": False,
            } for index in range(1, 4)]

    monkeypatch.setattr(main_module, "retriever", Retriever())
    monkeypatch.setattr(main_module, "workflow", build_workflow())
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/disruptions/analyze", json=payload, headers={"Authorization": f"Bearer {account['access_token']}"})

    assert response.status_code == 200
    assert response.json()["llm_status"] == "anthropic"
    assert response.json()["recommendation"]["recommended_response"] == "Alternative Supplier"
    assert len(response.json()["recommendation"]["summary"]) <= 600
    assert response.json()["recommendation"]["reasoning"] == ["This response is represented in the retrieved historical cases."]
    assert response.json()["provider_tracking"]["final_provider"] == "Anthropic Claude"
    assert response.json()["provider_tracking"]["final_model"] == "claude-sonnet-4-5"
    assert response.json()["provider_tracking"]["fallback_used"] is False
    assert response.json()["provider_tracking"]["provider_attempts"] == [{"provider": "anthropic", "model": "claude-sonnet-4-5", "status": "success"}]
    assert len(calls) >= 1
    assert "api.anthropic.com" in str(calls[0].url)
    terminal = capsys.readouterr().out
    assert "LLM provider: Anthropic Claude" in caplog.text
    assert "LLM model: claude-sonnet-4-5" in caplog.text
    assert "[LLM] Provider: Anthropic Claude | Model: claude-sonnet-4-5 | Status: SUCCESS" in caplog.text
    assert "test-anthropic-key" not in caplog.text + terminal
    history = request(api_app, "GET", "/api/history", headers={"Authorization": f"Bearer {account['access_token']}"})
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["final_provider"] == "Anthropic Claude"
    assert history.json()["items"][0]["final_model"] == "claude-sonnet-4-5"
    assert history.json()["items"][0]["fallback_used"] is False


def test_analysis_uses_anthropic_first_then_gemini_and_returns_200(api_app, monkeypatch, caplog, capsys):
    caplog.set_level(logging.INFO)
    account = register_user(api_app)
    calls = []
    response_body = {
        "summary": "Retrieved cases support an alternative supplier response. " + ("More context. " * 55),
        "recommended_response": "Alternative Supplier", "alternative_responses": [],
        "reasoning": "This response is represented in the retrieved historical cases.",
        "evidence": ["DIS-HISTORY-1", "DIS-HISTORY-2"], "confidence": 0.5, "human_review_required": True,
    }

    def provider_response(request):
        calls.append(request)
        if "api.anthropic.com" in str(request.url):
            assert "/v1/messages" in str(request.url)
            assert json.loads(request.content)["model"] == "claude-sonnet-4-5"
            return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "rate limited"}})
        assert "generativelanguage.googleapis.com" in str(request.url)
        assert "/models/gemini-3.5-flash-lite:generateContent" in str(request.url)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(response_body)}]}}]})

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_model", "claude-sonnet-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")

    class Retriever:
        def search(self, db, query, filters, top_k):
            return [{
                "disruption_id": f"DIS-HISTORY-{index}", "response_type": "Alternative Supplier",
                "production_impact_pct": 40.0 + index, "revenue_loss_usd": 1000.0 + index,
                "response_time_days": 4, "partial_recovery_days": 20, "full_recovery_days": 60,
                "permanent_supplier_change": False,
            } for index in range(1, 4)]

    monkeypatch.setattr(main_module, "retriever", Retriever())
    monkeypatch.setattr(main_module, "workflow", build_workflow())
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/disruptions/analyze", json=payload, headers={"Authorization": f"Bearer {account['access_token']}"})

    assert response.status_code == 200
    assert response.json()["llm_status"] == "gemini"
    assert response.json()["recommendation"]["recommended_response"] == "Alternative Supplier"
    assert len(response.json()["recommendation"]["summary"]) <= 600
    assert response.json()["recommendation"]["reasoning"] == ["This response is represented in the retrieved historical cases."]
    assert response.json()["provider_tracking"]["final_provider"] == "Google Gemini"
    assert response.json()["provider_tracking"]["final_model"] == "gemini-3.5-flash-lite"
    assert response.json()["provider_tracking"]["fallback_used"] is True
    assert [attempt["status"] for attempt in response.json()["provider_tracking"]["provider_attempts"]] == ["failed", "success"]
    assert len(calls) >= 2
    assert "api.anthropic.com" in str(calls[0].url)
    assert "generativelanguage.googleapis.com" in str(calls[1].url)
    terminal = capsys.readouterr().out
    assert "LLM provider: Anthropic Claude" in caplog.text
    assert "LLM model: claude-sonnet-4-5" in caplog.text
    assert "LLM PROVIDER FAILURE" in terminal
    assert "Status Code : 429" in terminal
    assert "Error       : HTTP 429: rate limited" in terminal
    assert "Anthropic Claude failed -> Starting fallback with Google Gemini" in terminal
    assert "LLM provider: Google Gemini" in caplog.text
    assert "LLM model: gemini-3.5-flash-lite" in caplog.text
    assert "[LLM] Provider: Google Gemini | Model: gemini-3.5-flash-lite | Status: SUCCESS" in caplog.text
    assert "test-anthropic-key" not in caplog.text + terminal
    assert "test-gemini-key" not in caplog.text + terminal
    history = request(api_app, "GET", "/api/history", headers={"Authorization": f"Bearer {account['access_token']}"})
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["final_provider"] == "Google Gemini"
    assert history.json()["items"][0]["final_model"] == "gemini-3.5-flash-lite"
    assert history.json()["items"][0]["fallback_used"] is True


def test_analysis_uses_deterministic_fallback_if_both_providers_fail(api_app, monkeypatch):
    account = register_user(api_app)
    calls = []

    def provider_response(request):
        calls.append(request)
        if "api.anthropic.com" in str(request.url):
            return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "rate limited"}})
        return httpx.Response(503, json={"error": "temporarily unavailable"})

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_model", "claude-sonnet-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")

    class Retriever:
        def search(self, db, query, filters, top_k):
            return []

    monkeypatch.setattr(main_module, "retriever", Retriever())
    monkeypatch.setattr(main_module, "workflow", build_workflow())
    payload = {
        "description": "A severe port disruption affects a regional electronics supplier.",
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/disruptions/analyze", json=payload, headers={"Authorization": f"Bearer {account['access_token']}"})

    assert response.status_code == 200
    assert len(calls) >= 2
    assert "api.anthropic.com" in str(calls[0].url)
    assert "generativelanguage.googleapis.com" in str(calls[1].url)
    assert response.json()["llm_status"] == "LLM unavailable; historical-data recommendation used"
    assert response.json()["recommendation"]["recommended_response"] == "Insufficient historical evidence."
    assert response.json()["provider_tracking"]["final_provider"] == "Deterministic Fallback"
    assert response.json()["provider_tracking"]["final_model"] == "extractive"
    assert response.json()["provider_tracking"]["fallback_used"] is True
    assert [attempt["status"] for attempt in response.json()["provider_tracking"]["provider_attempts"]] == ["failed", "failed"]
    history = request(api_app, "GET", "/api/history", headers={"Authorization": f"Bearer {account['access_token']}"})
    assert history.json()["items"][0]["final_provider"] == "Deterministic Fallback"
    assert history.json()["items"][0]["final_model"] == "extractive"


@pytest.mark.parametrize(("provider", "model"), [("anthropic", "claude-sonnet-4-5"), ("gemini", "gemini-3.5-flash-lite"), ("ollama", "llama3.2")])
def test_provider_success_tracking_uses_actual_model(monkeypatch, provider, model):
    calls = []

    def provider_response(request):
        calls.append(request)
        if provider == "anthropic":
            assert request.url.host == "api.anthropic.com"
            assert request.headers.get("x-api-key") == "test-anthropic-key"
            assert request.headers.get("anthropic-version") == "2023-06-01"
            assert json.loads(request.content)["model"] == model
            return httpx.Response(200, json={
                "id": "msg_test", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": "generated"}],
            })
        if provider == "gemini":
            assert request.url.path.endswith(f"/models/{model}:generateContent")
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "generated"}]}}]})
        assert request.url.host == "localhost"
        return httpx.Response(200, json={"choices": [{"message": {"content": "generated"}}]})

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))
    monkeypatch.setattr(settings, "llm_provider", provider)
    monkeypatch.setattr(settings, "llm_model", model)
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")

    completion = llm_module.complete_with_provider([{"role": "user", "content": "test"}])
    assert completion["content"] == "generated"
    assert completion["provider"] == provider
    assert completion["model"] == model
    assert completion["fallback_used"] is False
    assert completion["provider_attempts"] == [{"provider": provider, "model": model, "status": "success"}]
    assert len(calls) == 1


def test_retrieval_forwards_top_k_and_returns_empty_results(api_app, monkeypatch):
    class EmptyRetriever:
        def search(self, db, query, filters, top_k):
            assert top_k == 3
            return []

    monkeypatch.setattr(main_module, "retriever", EmptyRetriever())
    response = request(api_app, "POST", "/api/retrieval/similar", json={"query": "supplier disruption in asia", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert response.json()["count"] == 0


def test_retrieval_rejects_empty_query(api_app):
    response = request(api_app, "POST", "/api/retrieval/similar", json={"query": ""})
    assert response.status_code == 422


def test_retrieval_returns_service_unavailable_without_index(api_app, monkeypatch):
    monkeypatch.setattr(main_module, "retriever", None)
    response = request(api_app, "POST", "/api/retrieval/similar", json={"query": "supplier disruption"})
    assert response.status_code == 503


def test_prediction_api_loads_existing_serialized_models(api_app):
    available = initialize_models()
    assert all(available.values())
    payload = {
        "disruption_type": "Port Congestion", "industry": "Electronics", "supplier_tier": "2",
        "supplier_region": "Asia-Pacific", "supplier_size": "Medium", "has_backup_supplier": False,
        "disruption_severity": 4,
    }
    response = request(api_app, "POST", "/api/predictions", json=payload)
    assert response.status_code == 200
    assert set(response.json()["predictions"]) == {
        "production_impact", "financial_loss", "recovery_time", "permanent_supplier_change",
    }


def test_saved_recommendation_api_remains_available(api_app, session_factory):
    account = register_user(api_app)
    with session_factory.begin() as db:
        analysis = AnalysisRequest(request_data={"description": "test"})
        db.add(analysis)
        db.flush()
        db.add(Recommendation(analysis_id=analysis.id, recommendation_data={"recommended_response": "Alternative Supplier"}))
        analysis_id = analysis.id
        db.add(AnalysisHistory(
            user_id=account["user"]["id"], analysis_request_id=analysis.id, query_text="test", search_text="test",
            request_data={"description": "test"}, response_data={"recommendation": {"recommended_response": "Alternative Supplier"}},
            disruption_type="Port Congestion", industry="Electronics", supplier_region="Asia-Pacific", supplier_tier="2",
            severity=4, final_provider="Deterministic Fallback", final_model="extractive", fallback_used=False,
            provider_attempts=[], response_time_ms=100, status="success",
        ))
    response = request(api_app, "POST", f"/api/recommendations?analysis_id={analysis_id}", headers={"Authorization": f"Bearer {account['access_token']}"})
    assert response.status_code == 200
    assert response.json()["recommended_response"] == "Alternative Supplier"


def test_database_failure_returns_cors_enabled_service_unavailable(api_app):
    def unavailable_db():
        raise SQLAlchemyError("simulated database failure")
        yield  # pragma: no cover

    api_app.dependency_overrides.clear()
    api_app.dependency_overrides[main_module.get_db] = unavailable_db
    response = request(api_app, "GET", "/api/disruptions", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 503
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert response.json() == {"detail": "Database unavailable. Check DATABASE_URL and confirm local MySQL is running."}

