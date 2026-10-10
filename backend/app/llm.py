from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from urllib.parse import quote, quote_plus

import httpx

from app.config import settings

logger = logging.getLogger("uvicorn.error")
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"
PROVIDER_LABELS = {
    "anthropic": "Anthropic Claude",
    "gemini": "Google Gemini",
    "openai": "OpenAI",
    "ollama": "Ollama",
    "deterministic": "Deterministic Fallback",
}


class LLMUnavailable(RuntimeError):
    pass


class ProviderFailure(LLMUnavailable):
    def __init__(self, reason: str, status_code: int | None = None):
        super().__init__(reason)
        self.status_code = status_code


def _configured_key(value: str) -> bool:
    return bool(value and value.strip() and not value.strip().upper().startswith("YOUR_"))


def _redact_provider_error(message: str) -> str:
    """Remove credentials and common token formats before writing provider errors to logs."""
    safe_message = message
    for key in (settings.anthropic_api_key, settings.gemini_api_key, settings.openai_api_key):
        if key and _configured_key(key):
            for secret in {key.strip(), quote(key.strip(), safe=""), quote_plus(key.strip())}:
                safe_message = safe_message.replace(secret, "[redacted]")
    safe_message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", safe_message)
    safe_message = re.sub(r"(?i)\b(?:x-goog-api-key|x-api-key|api[_-]?key|key)\s*[:=]\s*[^&\s,;]+", "API key=[redacted]", safe_message)
    safe_message = re.sub(r"\bsk-ant-[0-9A-Za-z_-]{20,}\b", "[redacted]", safe_message)
    safe_message = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[redacted]", safe_message)
    safe_message = re.sub(r"[\r\n\t]+", " ", safe_message)
    return safe_message[:240]


def _provider_failure_reason(exc: Exception) -> tuple[str, int | None]:
    """Describe a failed request without logging exception URLs, headers, or raw bodies."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        message = ""
        try:
            error = exc.response.json().get("error", {})
            if isinstance(error, dict):
                message = str(error.get("message", ""))
            elif isinstance(error, str):
                message = error
        except (ValueError, TypeError):
            pass
        reason = f"HTTP {status}"
        if message:
            reason += f": {_redact_provider_error(message)}"
        return reason, status
    if isinstance(exc, httpx.TimeoutException):
        return "request timed out", None
    if isinstance(exc, httpx.ConnectError):
        return "connection failed", None
    if isinstance(exc, httpx.HTTPError):
        return f"network request failed ({type(exc).__name__})", None
    if isinstance(exc, json.JSONDecodeError):
        return "provider returned invalid JSON", None
    if isinstance(exc, KeyError):
        return f"provider response is missing a required field ({_redact_provider_error(str(exc))})", None
    if isinstance(exc, IndexError):
        return "provider response contains no generated content", None
    if isinstance(exc, TypeError):
        return "provider response has an unexpected format", None
    return f"unexpected exception ({type(exc).__name__})", None


def _log_provider_start(provider: str, model: str) -> None:
    """Print a status banner when a request is about to be sent."""
    label = PROVIDER_LABELS.get(provider, provider)
    banner = "\n".join((
        "=" * 50,
        "LLM REQUEST START",
        f"Provider : {label}",
        f"Model    : {model}",
        f"Status   : Sending request ...",
        "=" * 50,
    ))
    print(banner, flush=True)


def _log_provider_success(provider: str, model: str, elapsed_ms: float) -> None:
    """Print a status banner when a request completes successfully."""
    label = PROVIDER_LABELS.get(provider, provider)
    banner = "\n".join((
        "=" * 50,
        "LLM REQUEST SUCCESS",
        f"Provider : {label}",
        f"Model    : {model}",
        f"Status   : OK",
        f"Latency  : {elapsed_ms:.0f} ms",
        "=" * 50,
    ))
    print(banner, flush=True)


def _log_provider_failure(provider: str, model: str, exc: Exception, action: str) -> None:
    """Print a status banner when a provider request fails."""
    label = PROVIDER_LABELS.get(provider, provider)
    status_code = getattr(exc, "status_code", None)
    status = str(status_code) if status_code is not None else "n/a"
    detail = str(exc)
    if status_code is not None:
        reason = "API request failed"
    elif "API key" in detail:
        reason = "Provider configuration error"
    elif "response" in detail.lower():
        reason = "Invalid or malformed provider response"
    elif "timed out" in detail.lower() or "connection" in detail.lower():
        reason = "Network request failed"
    elif detail.startswith("unexpected exception"):
        reason = "Unexpected exception"
    else:
        reason = "Provider request failed"
    detail = _redact_provider_error(detail)
    diagnostic = "\n".join((
        "=" * 50,
        "LLM PROVIDER FAILURE",
        f"Provider    : {label}",
        f"Model       : {model}",
        f"Reason      : {reason}",
        f"Status Code : {status}",
        f"Error       : {detail}",
        f"Action      : {action}",
        "=" * 50,
    ))
    print(diagnostic, flush=True)


def _content(response: httpx.Response, provider: str) -> str:
    data = response.json()
    if provider == "gemini":
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts)
    elif provider == "anthropic":
        parts = data.get("content", [])
        content = "".join(
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content:
        raise LLMUnavailable(f"The configured {provider} LLM returned a non-text response")
    return content


def _configured_model(provider: str) -> str:
    if provider == "anthropic":
        return settings.llm_model if settings.llm_provider.lower() == "anthropic" and settings.llm_model else (settings.anthropic_model or "claude-sonnet-4-5")
    if provider == "gemini":
        return settings.gemini_model or (settings.llm_model if settings.llm_provider.lower() == "gemini" and settings.llm_model else GEMINI_DEFAULT_MODEL)
    if provider == "openai":
        return settings.llm_model or "gpt-4o-mini"
    if provider == "ollama":
        return settings.llm_model or "llama3.1"
    return "configured"


def _request_provider(
    provider: str,
    messages: list[dict],
    *,
    json_mode: bool,
    temperature: float,
    timeout: float,
) -> tuple[str, str]:
    model = _configured_model(provider)
    logger.info("LLM provider: %s", PROVIDER_LABELS.get(provider, provider))
    logger.info("LLM model: %s", model)
    if provider == "anthropic":
        key = settings.anthropic_api_key
        if not _configured_key(key):
            raise LLMUnavailable("Anthropic API key is missing or still has its placeholder value")
        if settings.anthropic_endpoint.strip():
            endpoint = settings.anthropic_endpoint.strip()
        else:
            base_url = (settings.anthropic_base_url or "https://api.anthropic.com").strip().rstrip("/")
            if base_url.endswith("/v1/messages"):
                endpoint = base_url
            elif base_url.endswith("/v1"):
                endpoint = f"{base_url}/messages"
            else:
                endpoint = f"{base_url}/v1/messages"
        system_messages = [item["content"] for item in messages if item.get("role") == "system"]
        contents = [
            {"role": item["role"], "content": item["content"]}
            for item in messages if item.get("role") != "system"
        ]
        if not contents and system_messages:
            contents = [{"role": "user", "content": "\n\n".join(system_messages)}]
            system_messages = []
        payload = {
            "model": model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": contents,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    elif provider == "gemini":
        key = settings.gemini_api_key
        if not _configured_key(key):
            raise LLMUnavailable("Gemini API key is missing or still has its placeholder value")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        system_messages = [item["content"] for item in messages if item.get("role") == "system"]
        contents = [
            {"role": "model" if item.get("role") == "assistant" else "user", "parts": [{"text": item["content"]}]}
            for item in messages if item.get("role") != "system"
        ]
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_messages:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_messages)}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    elif provider == "openai":
        key = settings.openai_api_key
        if not _configured_key(key):
            raise LLMUnavailable("OpenAI API key is missing or still has its placeholder value")
        endpoint = "https://api.openai.com/v1/chat/completions"
        payload = {"model": model, "temperature": temperature, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    elif provider == "ollama":
        endpoint = settings.ollama_base_url.rstrip("/") + "/v1/chat/completions"
        payload = {"model": model, "temperature": temperature, "messages": messages}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {}
    else:
        raise LLMUnavailable(f"Unsupported LLM provider: {provider}")

    _log_provider_start(provider, model)
    _t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        elapsed_ms = (time.monotonic() - _t0) * 1000
        _log_provider_success(provider, model, elapsed_ms)
        return _content(response, provider), model
    except LLMUnavailable:
        raise
    except Exception as exc:
        # Never log the exception itself: HTTP errors can include credential-bearing URLs/headers.
        reason, status_code = _provider_failure_reason(exc)
        raise ProviderFailure(reason, status_code) from None


def complete_with_provider(
    messages: list[dict], *, json_mode: bool = False, temperature: float = 0.1, timeout: float = 30.0,
    validate_response: Callable[[str], str | None] | None = None,
) -> dict | None:
    """Call configured LLM providers, then leave final recommendation to deterministic workflow."""
    provider = (settings.llm_provider or "anthropic").lower()
    if provider == "demo":
        logger.info("[LLM] Using deterministic fallback (LLM_PROVIDER=demo).")
        demo_banner = "\n".join((
            "=" * 50,
            "LLM REQUEST SKIPPED",
            "Provider : demo (no LLM configured)",
            "Model    : n/a",
            "Status   : Using deterministic / rule-based fallback",
            "=" * 50,
        ))
        print(demo_banner, flush=True)
        return None

    if provider == "anthropic":
        providers = ["anthropic", "gemini"]
    elif provider == "gemini":
        providers = ["gemini"]
    else:
        providers = [provider]

    attempts = []
    for index, attempt in enumerate(providers):
        try:
            content, model = _request_provider(
                attempt, messages, json_mode=json_mode, temperature=temperature, timeout=timeout
            )
            if validate_response is not None:
                normalized = validate_response(content)
                if normalized is not None:
                    content = normalized
            attempts.append({"provider": attempt, "model": model, "status": "success"})
            logger.info(
                "[LLM] Provider: %s | Model: %s | Status: SUCCESS",
                PROVIDER_LABELS.get(attempt, attempt), model,
            )
            return {
                "content": content, "provider": attempt, "model": model,
                "fallback_used": index > 0, "provider_attempts": attempts,
            }
        except Exception as exc:
            model = _configured_model(attempt)
            attempts.append({"provider": attempt, "model": model, "status": "failed"})
            label = PROVIDER_LABELS.get(attempt, attempt)
            if index + 1 < len(providers):
                next_label = PROVIDER_LABELS.get(providers[index + 1], providers[index + 1])
                action = f"Switching to fallback provider: {next_label}"
            else:
                action = "All providers exhausted — switching to deterministic analysis"
            _log_provider_failure(attempt, model, exc, action)
            logger.warning("[LLM] Provider %s (%s) failed: %s. %s", label, model, _redact_provider_error(str(exc)), action)
            if index + 1 < len(providers):
                next_label = PROVIDER_LABELS.get(providers[index + 1], providers[index + 1])
                print(f"[LLM] {label} failed -> Starting fallback with {next_label}", flush=True)
            else:
                print(f"[LLM] {label} failed -> Starting deterministic/local fallback", flush=True)
    if attempts:
        print("\n".join((
            "=" * 50,
            "LLM FALLBACK ACTIVE",
            "Status  : All configured LLM providers failed",
            "Action  : Using deterministic / rule-based analysis",
            "=" * 50,
        )), flush=True)
    return {
        "content": None, "provider": "deterministic", "model": "extractive",
        "fallback_used": bool(attempts), "provider_attempts": attempts,
    }


def chat_completion(
    messages: list[dict], *, json_mode: bool = False, temperature: float = 0.1, timeout: float = 30.0
) -> str | None:
    """Compatibility wrapper for callers that only need generated text."""
    result = complete_with_provider(messages, json_mode=json_mode, temperature=temperature, timeout=timeout)
    return result["content"] if result else None
