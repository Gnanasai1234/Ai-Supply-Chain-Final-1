"""Offline retrieval and recommendation-contract evaluation for the RAG pipeline.

Run from any directory with: python evaluation/evaluate.py
The benchmark uses the checked-in dataset, Chroma/BM25 artifacts, and a temporary
SQLite database. It does not need MySQL, an LLM API key, or a DeepEval service.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
# App models import the shared database module. Force its unused global engine to
# SQLite for this standalone command so a local MySQL driver is not required.
os.environ["DATABASE_URL"] = "sqlite://"

from sqlalchemy import create_engine, insert, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.models import Base, Disruption  # noqa: E402
from app.rag.documents import disruption_to_document  # noqa: E402
from app.rag.hybrid_retriever import HybridRetriever  # noqa: E402
from app.services.analysis import recommendation_from_history, response_analysis  # noqa: E402


CASE_FILE = ROOT / "evaluation" / "cases.json"
DATASET_FILE = ROOT / "backend" / "data" / "supply_chain_disruption_recovery.csv"
TOP_K = 10
THRESHOLDS = {
    "context_precision_at_10": 0.95,
    "labeled_recall_at_10": 0.75,
    "mean_reciprocal_rank": 0.75,
    "recommendation_support_rate": 1.0,
    "evidence_consistency_rate": 1.0,
    "insufficient_evidence_handling": 1.0,
}


class DatasetLexicalFallback:
    """Small transparent baseline used only when local RAG artifacts are unavailable."""

    @staticmethod
    def search(db: Session, query: str, filters: dict, top_k: int = TOP_K) -> list[dict]:
        statement = HybridRetriever._db_statement(filters)
        rows = db.scalars(statement).all()
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        explicit_ids = {value.lower() for value in re.findall(r"DIS-\d+", query, flags=re.IGNORECASE)}
        ranked = []
        for row in rows:
            document = disruption_to_document(row).lower()
            score = sum(document.count(token) for token in tokens)
            if row.disruption_id.lower() in explicit_ids:
                score += 100
            ranked.append((score, row))
        ranked.sort(key=lambda item: (-item[0], item[1].disruption_id))
        return [
            {
                "disruption_id": row.disruption_id,
                "disruption_type": row.disruption_type,
                "industry": row.industry,
                "supplier_tier": row.supplier_tier,
                "supplier_region": row.supplier_region,
                "supplier_size": row.supplier_size,
                "has_backup_supplier": row.has_backup_supplier,
                "disruption_severity": row.disruption_severity,
                "production_impact_pct": row.production_impact_pct,
                "revenue_loss_usd": row.revenue_loss_usd,
                "response_type": row.response_type,
                "response_time_days": row.response_time_days,
                "partial_recovery_days": row.partial_recovery_days,
                "full_recovery_days": row.full_recovery_days,
                "permanent_supplier_change": row.permanent_supplier_change,
            }
            for _, row in ranked[:top_k]
        ]


def load_cases() -> list[dict]:
    return json.loads(CASE_FILE.read_text(encoding="utf-8"))


def load_dataset_into_sqlite(engine) -> int:
    """Load the source CSV into a private in-memory DB for the real retriever."""
    count = 0
    columns = [column.name for column in Disruption.__table__.columns if column.name not in {"id", "created_at"}]
    with DATASET_FILE.open(newline="", encoding="utf-8-sig") as source, engine.begin() as connection:
        reader = csv.DictReader(source)
        batch = []
        for row in reader:
            values = {name: row[name] for name in columns}
            values["supplier_tier"] = str(values["supplier_tier"])
            values["has_backup_supplier"] = values["has_backup_supplier"].strip().lower() == "true"
            values["permanent_supplier_change"] = values["permanent_supplier_change"].strip().lower() == "true"
            values["disruption_severity"] = int(values["disruption_severity"])
            for name in ("production_impact_pct", "revenue_loss_usd"):
                values[name] = float(values[name])
            for name in ("response_time_days", "partial_recovery_days", "full_recovery_days"):
                values[name] = int(values[name])
            batch.append(values)
            if len(batch) >= 5000:
                connection.execute(insert(Disruption), batch)
                count += len(batch)
                batch.clear()
        if batch:
            connection.execute(insert(Disruption), batch)
            count += len(batch)
    return count


def case_relevant(result: dict, filters: dict) -> bool:
    for name, expected in filters.items():
        observed = result.get(name)
        if name == "disruption_severity" and isinstance(expected, dict):
            if "$gte" in expected and observed < expected["$gte"]:
                return False
            if "$lte" in expected and observed > expected["$lte"]:
                return False
            if "$eq" in expected and observed != expected["$eq"]:
                return False
        elif str(observed) != str(expected):
            return False
    return True


def measure_case(case: dict, retriever: HybridRetriever, db: Session) -> dict:
    results = retriever.search(db, case["query"], case["filters"], top_k=TOP_K)
    relevant_ids = set(case["relevant_ids"])
    returned_ids = [row["disruption_id"] for row in results]
    relevant_ranks = [index for index, row_id in enumerate(returned_ids, 1) if row_id in relevant_ids]
    recall = len(set(returned_ids) & relevant_ids) / len(relevant_ids) if relevant_ids else None
    precision = (
        sum(case_relevant(row, case["filters"]) for row in results) / len(results)
        if results else (1.0 if case.get("expect_empty") else 0.0)
    )

    groups = response_analysis(results)
    recommendation = recommendation_from_history(groups, len(results))
    allowed = {group["response_type"] for group in groups}
    if len(results) < 3:
        recommendation_supported = (
            recommendation["recommended_response"] == "Insufficient historical evidence."
            and recommendation["human_review_required"] is True
        )
        evidence_consistent = recommendation_supported
    else:
        selected = next((group for group in groups if group["response_type"] == recommendation["recommended_response"]), None)
        recommendation_supported = recommendation["recommended_response"] in allowed
        evidence_consistent = bool(selected) and all(
            fragment in " ".join(recommendation["reasoning"])
            for fragment in (
                f"{selected['median_full_recovery_days']} days",
                f"{selected['median_production_impact_pct']}%",
                f"${selected['median_revenue_loss_usd']:,.2f}",
            )
        ) and set(selected["disruption_ids"]).issubset(set(returned_ids))

    expect_empty_ok = not case.get("expect_empty") or not results
    reference_found = not relevant_ids or bool(set(returned_ids) & relevant_ids)
    return {
        "name": case["name"],
        "retrieved": len(results),
        "retrieved_ids": returned_ids,
        "precision": precision,
        "recall": recall,
        "reciprocal_rank": 1 / relevant_ranks[0] if relevant_ranks else (0.0 if relevant_ids else None),
        "recommendation_supported": recommendation_supported,
        "evidence_consistent": evidence_consistent,
        "insufficient_evidence_handling": expect_empty_ok,
        "reference_found": reference_found,
        "case_status": "PASS" if reference_found and recommendation_supported and evidence_consistent and expect_empty_ok else "FAIL",
        "empty_expected": bool(case.get("expect_empty")),
    }


def metric(name: str, score: float | None, threshold: float | None = None, *, note: str = "") -> dict:
    if score is None:
        return {"name": name, "score": None, "threshold": threshold, "status": "NOT EVALUATED", "note": note}
    passed = threshold is None or score >= threshold
    return {"name": name, "score": score, "threshold": threshold, "status": "PASS" if passed else "FAIL", "note": note}


def main() -> int:
    if not DATASET_FILE.exists():
        print(f"Dataset not found: {DATASET_FILE}", file=sys.stderr)
        return 2

    cases = load_cases()
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    try:
        Base.metadata.create_all(engine)
        dataset_rows = load_dataset_into_sqlite(engine)
        try:
            retriever = HybridRetriever()
            evaluation_mode = "application hybrid retriever (Chroma + BM25)"
        except Exception as exc:
            retriever = DatasetLexicalFallback()
            evaluation_mode = f"dataset lexical fallback; hybrid retriever unavailable ({type(exc).__name__}: {exc})"
        with Session(engine) as db:
            case_results = [measure_case(case, retriever, db) for case in cases]

        labeled_cases = [row for row in case_results if row["recall"] is not None]
        metrics = [
            metric("Context precision@10", mean(row["precision"] for row in case_results if row["retrieved"] or not row["empty_expected"]), THRESHOLDS["context_precision_at_10"], note="Share of retrieved contexts satisfying the test case's labeled business filters."),
            metric("Labeled recall@10", mean(row["recall"] for row in labeled_cases), THRESHOLDS["labeled_recall_at_10"], note="Share of hand-labeled reference cases retrieved."),
            metric("Mean reciprocal rank", mean(row["reciprocal_rank"] for row in labeled_cases), THRESHOLDS["mean_reciprocal_rank"], note="Rank of the first hand-labeled reference case."),
            metric("Recommendation support rate", mean(float(row["recommendation_supported"]) for row in case_results), THRESHOLDS["recommendation_support_rate"], note="Selected strategy belongs to the retrieved historical response groups; sparse evidence triggers review."),
            metric("Evidence consistency rate", mean(float(row["evidence_consistent"]) for row in case_results), THRESHOLDS["evidence_consistency_rate"], note="Deterministic rationale values and cited case IDs match retrieved evidence."),
            metric("Empty-result handling", mean(float(row["insufficient_evidence_handling"]) for row in case_results if row["empty_expected"]), THRESHOLDS["insufficient_evidence_handling"], note="No-match query yields no cases and an insufficient-evidence recommendation."),
            metric("DeepEval faithfulness", None, note="LLM judge not run by this offline command. Existing optional judge: python evaluation/run_evaluation.py (requires configured provider and saved analyses)."),
            metric("DeepEval answer relevancy", None, note="LLM judge not run by this offline command. Existing optional judge: python evaluation/run_evaluation.py (requires configured provider and saved analyses)."),
            metric("DeepEval contextual relevancy", None, note="LLM judge not run by this offline command. Existing optional judge: python evaluation/run_evaluation.py (requires configured provider and saved analyses)."),
        ]
        required = [item for item in metrics if item["threshold"] is not None]
        overall_pass = all(item["status"] == "PASS" for item in required)
        report = {
            "dataset": str(DATASET_FILE.relative_to(ROOT)),
            "dataset_rows": dataset_rows,
            "evaluation_mode": evaluation_mode,
            "test_cases": len(cases),
            "top_k": TOP_K,
            "overall": "PASS" if overall_pass else "FAIL",
            "llm_quality_evaluation": "NOT EVALUATED",
            "metrics": metrics,
            "cases": case_results,
        }
        report_path = ROOT / "evaluation" / "reports" / "latest_offline_evaluation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"Offline RAG evaluation — {dataset_rows:,} dataset rows, {len(cases)} cases, k={TOP_K}")
        print(f"Mode: {evaluation_mode}")
        print(f"{'Metric':34} {'Score':>8}  {'Result':<13} Details")
        print("-" * 108)
        for item in metrics:
            score = "—" if item["score"] is None else f"{item['score']:.3f}"
            print(f"{item['name']:34} {score:>8}  {item['status']:<13} {item['note']}")
        print("-" * 108)
        print(f"Overall local evaluation: {report['overall']} (LLM answer-quality metrics: NOT EVALUATED)")
        print("\nCase results:")
        for row in case_results:
            print(f"  {row['case_status']}: {row['name']} ({row['retrieved']} retrieved)")
        print(f"Detailed report: {report_path}")
        return 0 if overall_pass else 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
