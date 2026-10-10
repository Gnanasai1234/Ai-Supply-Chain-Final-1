import pytest
from app.evaluation_service import calculate_overall_evaluation, evaluate_ragas_metrics


def test_full_analysis_evaluation_flow():
    """
    Simulates the exact end-to-end evaluation flow:
    User Input -> Retrieval -> Analysis/Prediction -> LLM Response -> RAGAS Evaluation -> RAGAS Scores
    """
    # 1. User Input
    user_input = {
        "disruption_type": "Port Congestion",
        "industry": "Electronics",
        "supplier_region": "Asia-Pacific",
        "supplier_tier": "2",
        "supplier_size": "Medium",
        "has_backup_supplier": True,
        "disruption_severity": 4,
        "description": "Port congestion halting container transshipments from key Asia-Pacific terminal",
    }

    # 2. Retrieved context (from RAG / HybridRetriever)
    retrieved_cases = [
        {
            "disruption_id": "DIS-0101",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_region": "Asia-Pacific",
            "supplier_tier": "2",
            "disruption_severity": 4,
            "response_type": "Carrier Rerouting & Safety Stock",
            "full_recovery_days": 32,
            "similarity_score": 0.89,
        },
        {
            "disruption_id": "DIS-0102",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_region": "Asia-Pacific",
            "supplier_tier": "2",
            "disruption_severity": 4,
            "response_type": "Carrier Rerouting & Safety Stock",
            "full_recovery_days": 28,
            "similarity_score": 0.82,
        },
    ]

    # 3. Generated LLM Response / Recommendation
    recommendation = {
        "recommended_response": "Carrier Rerouting & Safety Stock",
        "summary": "Reroute cargo through adjacent regional terminals and release safety inventory.",
        "recommended_actions": [
            "Contract secondary shipping lines for urgent transshipments",
            "Release 15 days of buffer stock to maintain factory throughput",
            "Establish daily status synchronization with port logistics coordinators",
        ],
        "reasoning": [
            "Historical cases indicate 30-day recovery when rerouting is combined with safety stock.",
        ],
        "human_review_required": True,
    }

    # 4. RAGAS Evaluation (The ONLY evaluation method)
    eval_report = calculate_overall_evaluation(
        input_data=user_input,
        recommendation=recommendation,
        retrieved_cases=retrieved_cases,
        threshold=75.0,
    )

    # 5. Verify RAGAS is the ONLY evaluation method
    assert eval_report["evaluation_method"] == "RAGAS"
    assert "retrieval" not in eval_report
    assert "deepeval" not in eval_report
    assert "test_cases" not in eval_report
    assert "weights" not in eval_report

    # 6. Verify real RAGAS scores returned
    ragas = eval_report["ragas"]
    assert ragas["status"] == "available"
    assert ragas["framework"] == "RAGAS"
    assert eval_report["overall_score_pct"] == ragas["overall_pct"]
    assert 0.0 <= ragas["overall_score"] <= 1.0

    metrics = ragas["metrics"]
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        assert m in metrics
        metric_data = metrics[m]
        assert "score" in metric_data
        assert "score_pct" in metric_data
        assert "passed" in metric_data
        assert "status" in metric_data
        assert "reason" in metric_data
        assert 0.0 <= metric_data["score"] <= 1.0

    # Faithfulness should reflect grounding in retrieved response type
    assert metrics["faithfulness"]["score"] > 0.0
    # Context precision should reflect top ranking of matching cases
    assert metrics["context_precision"]["score"] > 0.0
    # Context recall should reflect facet coverage
    assert metrics["context_recall"]["score"] > 0.0

    print("\nVerified RAGAS evaluation flow successfully:")
    print(f"Overall RAGAS Score: {eval_report['overall_score_pct']}% ({eval_report['status']})")
    print(f"Faithfulness       : {metrics['faithfulness']['score_pct']}%")
    print(f"Answer Relevancy   : {metrics['answer_relevancy']['score_pct']}%")
    print(f"Context Precision  : {metrics['context_precision']['score_pct']}%")
    print(f"Context Recall     : {metrics['context_recall']['score_pct']}%")
