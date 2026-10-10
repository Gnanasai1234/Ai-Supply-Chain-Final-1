import sys
import os
import json
sys.path.insert(0, os.path.abspath("backend"))

from app.database import SessionLocal
from app.models import Disruption, AnalysisRequest
from app.agents.workflow import build_workflow
from app.rag.hybrid_retriever import HybridRetriever
from app.evaluation_service import calculate_overall_evaluation, log_evaluation_console
from app.config import settings

print("=== INITIALIZING RETRIEVER & WORKFLOW ===")
retriever = HybridRetriever()
workflow = build_workflow()

scenarios = [
    {
        "name": "TEST 1: Port Congestion / Electronics / Asia-Pacific / Severity 4",
        "input": {
            "description": "Severe congestion at major regional container terminals has halted electronics component shipments for over two weeks.",
            "disruption_type": "Port Congestion",
            "industry": "Electronics",
            "supplier_tier": "2",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Medium",
            "has_backup_supplier": False,
            "disruption_severity": 4,
        }
    },
    {
        "name": "TEST 2: Cyber Attack / Electronics / Asia-Pacific",
        "input": {
            "description": "Ransomware breach compromised supplier inventory ERP and operations systems, causing immediate production stoppage.",
            "disruption_type": "Cyber Attack",
            "industry": "Electronics",
            "supplier_tier": "1",
            "supplier_region": "Asia-Pacific",
            "supplier_size": "Large",
            "has_backup_supplier": False,
            "disruption_severity": 3,
        }
    },
    {
        "name": "TEST 3: Natural Disaster / Automotive / North America",
        "input": {
            "description": "Catastrophic hurricane disrupted transport networks and flooded tier-1 automotive manufacturing plants.",
            "disruption_type": "Natural Disaster",
            "industry": "Automotive",
            "supplier_tier": "1",
            "supplier_region": "North America",
            "supplier_size": "Medium",
            "has_backup_supplier": False,
            "disruption_severity": 4,
        }
    }
]

with SessionLocal() as db:
    for idx, sc in enumerate(scenarios, 1):
        print("\n" + "=" * 70)
        print(f"RUNNING SCENARIO {idx}: {sc['name']}")
        print("=" * 70)
        
        inp = sc["input"]
        
        # 1. Trace Query construction & Hybrid Retrieval
        query = f"{inp['description']} {inp['disruption_type']} {inp['industry']} Tier {inp['supplier_tier']} {inp['supplier_region']} {inp['supplier_size']}"
        filters = {
            "industry": inp["industry"],
            "supplier_tier": str(inp["supplier_tier"]),
            "supplier_region": inp["supplier_region"],
            "disruption_severity": {"$gte": max(1, inp["disruption_severity"] - 1)}
        }
        
        # Chroma count
        pool_size = 100
        vector = retriever.encoder.encode([query], normalize_embeddings=True).astype("float32")[0].tolist()
        chroma_res = retriever.collection.query(
            query_embeddings=[vector],
            n_results=pool_size,
            where=retriever._where(filters),
            include=["metadatas", "distances"]
        )
        chroma_ids = (chroma_res.get("ids") or [[]])[0]
        
        # BM25 count
        import re
        import numpy as np
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        all_scores = np.asarray(retriever.bm25.get_scores(tokens), dtype=float)
        keyword_indexes = np.argpartition(all_scores, -pool_size)[-pool_size:] if pool_size < len(all_scores) else np.arange(len(all_scores))
        bm25_count = len(keyword_indexes)
        
        # Execute workflow
        state = {
            "user_input": inp,
            "validated_input": inp,
            "db": db,
            "retriever": retriever,
            "priority_reference": {},
        }
        result = workflow.invoke(state)
        
        retrieved_cases = result.get("retrieved_cases", [])
        recommendation = result.get("recommendations", {})
        
        print("\n" + "-" * 50)
        print("RETRIEVAL PIPELINE TRACE")
        print("-" * 50)
        print(f"Query                   : {query}")
        print(f"Chroma result count     : {len(chroma_ids)}")
        print(f"BM25 result count       : {bm25_count}")
        print(f"Hybrid result count     : {len(retrieved_cases)}")
        print(f"Final context count     : {len(retrieved_cases)}")
        
        print("\n" + "-" * 50)
        print("GENERATED ANSWER")
        print("-" * 50)
        print(f"Recommended Response    : {recommendation.get('recommended_response')}")
        print(f"Summary                 : {recommendation.get('summary')}")
        print(f"Reasoning               : {recommendation.get('reasoning')}")
        
        print("\n" + "-" * 50)
        print("RAGAS METRICS & OVERALL SCORE")
        print("-" * 50)
        eval_report = calculate_overall_evaluation(
            input_data=inp,
            recommendation=recommendation,
            retrieved_cases=retrieved_cases,
        )
        ragas = eval_report.get("ragas", {})
        metrics = ragas.get("metrics", {})
        
        print(f"Faithfulness            : {metrics.get('faithfulness', {}).get('score_pct', 0)}%")
        print(f"Answer Relevancy        : {metrics.get('answer_relevancy', {}).get('score_pct', 0)}%")
        print(f"Context Precision       : {metrics.get('context_precision', {}).get('score_pct', 0)}%")
        print(f"Context Recall          : {metrics.get('context_recall', {}).get('score_pct', 0)}%")
        print(f"Overall RAGAS Score     : {eval_report.get('overall_score_pct', 0)}%")
        
        log_evaluation_console(eval_report, inp)
