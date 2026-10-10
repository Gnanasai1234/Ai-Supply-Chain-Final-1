from typing import Any, TypedDict


class SupplyChainState(TypedDict, total=False):
    user_input: dict
    validated_input: dict
    db: Any
    retriever: Any
    priority_reference: dict
    classification: dict
    investigation: dict
    retrieved_cases: list[dict]
    predictions: dict
    impact_analysis: dict
    response_analysis: list[dict]
    recovery_analysis: dict
    resilience_analysis: dict
    priority: dict
    recommendations: dict
    evidence: list[dict]
    explanation: dict
    validation_result: dict
    llm_status: str
    provider_tracking: dict
    what_if_simulation: list[dict]
