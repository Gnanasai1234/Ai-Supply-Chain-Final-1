from __future__ import annotations

import numpy as np

from app.rag.hybrid_retriever import HybridRetriever
from app.models import Disruption


def add_case(db, disruption_id, disruption_type="Port Congestion"):
    db.add(Disruption(
        disruption_id=disruption_id, disruption_type=disruption_type, industry="Electronics",
        supplier_tier="2", supplier_region="Asia-Pacific", supplier_size="Medium",
        has_backup_supplier=False, disruption_severity=4, production_impact_pct=54.3,
        revenue_loss_usd=1000000, response_type="Alternative Supplier", response_time_days=4,
        partial_recovery_days=20, full_recovery_days=61, permanent_supplier_change=False,
    ))


def test_retrieval_metadata_filter_compiles_to_chroma_and_sql_filters():
    filters = {"industry": "Electronics", "supplier_tier": "2", "disruption_severity": {"$gte": 3, "$lte": 5}}
    where = HybridRetriever._where(filters)
    assert where == {"$and": [
        {"industry": {"$eq": "Electronics"}},
        {"supplier_tier": {"$eq": "2"}},
        {"disruption_severity": {"$gte": 3}},
        {"disruption_severity": {"$lte": 5}},
    ]}


def test_unknown_metadata_fields_are_ignored():
    assert HybridRetriever._where({"not_a_filter": "value"}) is None


def test_search_ranks_results_and_applies_top_k(session_factory):
    class Encoder:
        def encode(self, queries, normalize_embeddings):
            return np.asarray([[1.0, 0.0]], dtype="float32")

    class Collection:
        def query(self, **kwargs):
            return {"ids": [["DIS-TEST-0001", "DIS-TEST-0002"]], "distances": [[0.2, 1.2]]}

    class BM25:
        def get_scores(self, tokens):
            return np.asarray([2.0, 1.0])

    retriever = object.__new__(HybridRetriever)
    retriever.encoder = Encoder()
    retriever.collection = Collection()
    retriever.bm25 = BM25()
    retriever.bm25_ids = ["DIS-TEST-0001", "DIS-TEST-0002"]
    with session_factory.begin() as db:
        add_case(db, "DIS-TEST-0001")
        add_case(db, "DIS-TEST-0002")
    with session_factory() as db:
        results = retriever.search(db, "supplier disruption", top_k=1)
    assert len(results) == 1
    assert results[0]["disruption_id"] == "DIS-TEST-0001"
    assert results[0]["semantic_score"] == 0.9
    assert results[0]["keyword_score"] == 1.0
    assert results[0]["hybrid_score"] == 0.93


def test_search_returns_empty_when_database_filters_remove_candidates(session_factory):
    class Encoder:
        def encode(self, queries, normalize_embeddings):
            return np.asarray([[1.0, 0.0]], dtype="float32")

    class Collection:
        def query(self, **kwargs):
            return {"ids": [["DIS-TEST-0001"]], "distances": [[0.2]]}

    class BM25:
        def get_scores(self, tokens):
            return np.asarray([1.0])

    retriever = object.__new__(HybridRetriever)
    retriever.encoder = Encoder()
    retriever.collection = Collection()
    retriever.bm25 = BM25()
    retriever.bm25_ids = ["DIS-TEST-0001"]
    with session_factory.begin() as db:
        add_case(db, "DIS-TEST-0001")
    with session_factory() as db:
        results = retriever.search(db, "supplier disruption", {"disruption_type": "Natural Disaster"}, top_k=5)
    assert results == []

