from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from app.config import settings
from app import llm as llm_module


@pytest.fixture(autouse=True)
def setup_llm_env(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_model", "claude-sonnet-4-5")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    monkeypatch.setattr(settings, "gemini_api_key", "test-gemini-key")
    monkeypatch.setattr(settings, "gemini_model", "gemini-3.5-flash-lite")


def test_path_1_anthropic_succeeds(monkeypatch, caplog, capsys):
    """Path 1: Primary provider Anthropic Claude succeeds."""
    caplog.set_level(logging.INFO)
    calls = []
    response_payload = {"recommended_response": "Alternative Supplier", "confidence": 0.8}

    def provider_response(request):
        calls.append(request)
        assert "api.anthropic.com" in str(request.url)
        assert request.headers.get("x-api-key") == "test-anthropic-key"
        assert request.headers.get("anthropic-version") == "2023-06-01"
        data = json.loads(request.content)
        assert data["model"] == "claude-sonnet-4-5"
        assert "messages" in data
        return httpx.Response(200, json={
            "id": "msg_001",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": json.dumps(response_payload)}],
        })

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))

    result = llm_module.complete_with_provider([
        {"role": "system", "content": "You are a supply chain assistant."},
        {"role": "user", "content": "Assess port disruption"},
    ], json_mode=True)

    assert result is not None
    assert result["provider"] == "anthropic"
    assert result["model"] == "claude-sonnet-4-5"
    assert result["fallback_used"] is False
    assert result["content"] == json.dumps(response_payload)
    assert result["provider_attempts"] == [
        {"provider": "anthropic", "model": "claude-sonnet-4-5", "status": "success"}
    ]
    assert len(calls) == 1

    terminal = capsys.readouterr().out
    assert "LLM provider: Anthropic Claude" in caplog.text
    assert "LLM model: claude-sonnet-4-5" in caplog.text
    assert "[LLM] Provider: Anthropic Claude | Model: claude-sonnet-4-5 | Status: SUCCESS" in caplog.text
    assert "test-anthropic-key" not in caplog.text + terminal


def test_path_2_anthropic_fails_and_gemini_succeeds(monkeypatch, caplog, capsys):
    """Path 2: Anthropic Claude fails (HTTP 429), then first fallback Google Gemini succeeds."""
    caplog.set_level(logging.INFO)
    calls = []
    response_payload = {"recommended_response": "Alternative Supplier", "confidence": 0.75}

    def provider_response(request):
        calls.append(request)
        if "api.anthropic.com" in str(request.url):
            return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "rate limit exceeded"}})
        if "generativelanguage.googleapis.com" in str(request.url):
            assert "/models/gemini-3.5-flash-lite:generateContent" in str(request.url)
            assert request.headers.get("x-goog-api-key") == "test-gemini-key"
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": json.dumps(response_payload)}]}}]
            })
        return httpx.Response(500, json={"error": "unexpected endpoint"})

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))

    result = llm_module.complete_with_provider([
        {"role": "system", "content": "You are a supply chain assistant."},
        {"role": "user", "content": "Assess port disruption"},
    ], json_mode=True)

    assert result is not None
    assert result["provider"] == "gemini"
    assert result["model"] == "gemini-3.5-flash-lite"
    assert result["fallback_used"] is True
    assert result["content"] == json.dumps(response_payload)
    assert result["provider_attempts"] == [
        {"provider": "anthropic", "model": "claude-sonnet-4-5", "status": "failed"},
        {"provider": "gemini", "model": "gemini-3.5-flash-lite", "status": "success"},
    ]
    assert len(calls) == 2

    terminal = capsys.readouterr().out
    assert "LLM provider: Anthropic Claude" in caplog.text
    assert "LLM PROVIDER FAILURE" in terminal
    assert "Status Code : 429" in terminal
    assert "Error       : HTTP 429: rate limit exceeded" in terminal
    assert "Anthropic Claude failed -> Starting fallback with Google Gemini" in terminal
    assert "LLM provider: Google Gemini" in caplog.text
    assert "LLM model: gemini-3.5-flash-lite" in caplog.text
    assert "[LLM] Provider: Google Gemini | Model: gemini-3.5-flash-lite | Status: SUCCESS" in caplog.text
    assert "test-anthropic-key" not in caplog.text + terminal
    assert "test-gemini-key" not in caplog.text + terminal


def test_path_3_both_fail_and_deterministic_fallback_succeeds(monkeypatch, caplog, capsys):
    """Path 3: Both Anthropic and Gemini fail, deterministic fallback is activated."""
    caplog.set_level(logging.INFO)
    calls = []

    def provider_response(request):
        calls.append(request)
        if "api.anthropic.com" in str(request.url):
            return httpx.Response(500, json={"error": {"type": "api_error", "message": "Anthropic internal server error"}})
        if "generativelanguage.googleapis.com" in str(request.url):
            return httpx.Response(503, json={"error": {"message": "Gemini temporarily unavailable"}})
        return httpx.Response(500, json={"error": "unexpected endpoint"})

    original_client = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(provider_response), **kwargs))

    result = llm_module.complete_with_provider([
        {"role": "system", "content": "You are a supply chain assistant."},
        {"role": "user", "content": "Assess port disruption"},
    ], json_mode=True)

    assert result is not None
    assert result["provider"] == "deterministic"
    assert result["model"] == "extractive"
    assert result["fallback_used"] is True
    assert result["content"] is None
    assert result["provider_attempts"] == [
        {"provider": "anthropic", "model": "claude-sonnet-4-5", "status": "failed"},
        {"provider": "gemini", "model": "gemini-3.5-flash-lite", "status": "failed"},
    ]
    assert len(calls) == 2

    terminal = capsys.readouterr().out
    assert "LLM provider: Anthropic Claude" in caplog.text
    assert "Anthropic Claude failed -> Starting fallback with Google Gemini" in terminal
    assert "LLM provider: Google Gemini" in caplog.text
    assert "Google Gemini failed -> Starting deterministic/local fallback" in terminal
    assert "LLM FALLBACK ACTIVE" in terminal
    assert "Status  : All configured LLM providers failed" in terminal
    assert "Action  : Using deterministic / rule-based analysis" in terminal
    assert "test-anthropic-key" not in caplog.text + terminal
    assert "test-gemini-key" not in caplog.text + terminal
