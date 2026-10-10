from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Disruption

FILTER_FIELDS = {
    "disruption_type": Disruption.disruption_type,
    "industry": Disruption.industry,
    "supplier_tier": Disruption.supplier_tier,
    "supplier_region": Disruption.supplier_region,
    "supplier_size": Disruption.supplier_size,
    "has_backup_supplier": Disruption.has_backup_supplier,
    "disruption_severity": Disruption.disruption_severity,
    "response_type": Disruption.response_type,
}
logger = logging.getLogger("uvicorn.error")


def normalize_filter_dict(filters: dict[str, Any] | None) -> dict[str, Any]:
    """
    Normalizes metadata filters to match dataset representations:
    Handles case normalization, whitespace, tier format, and severity numeric types.
    """
    if not filters:
        return {}

    canonical_disruptions = {
        "factory incident": "Factory Incident",
        "cyber attack": "Cyber Attack",
        "port congestion": "Port Congestion",
        "labor strike": "Labor Strike",
        "geopolitical": "Geopolitical",
        "natural disaster": "Natural Disaster",
    }
    canonical_industries = {
        "pharmaceuticals": "Pharmaceuticals",
        "pharma": "Pharmaceuticals",
        "electronics": "Electronics",
        "consumer goods": "Consumer Goods",
        "automotive": "Automotive",
        "auto": "Automotive",
        "aerospace": "Aerospace",
    }
    canonical_regions = {
        "asia-pacific": "Asia-Pacific",
        "asia pacific": "Asia-Pacific",
        "apac": "Asia-Pacific",
        "europe": "Europe",
        "eu": "Europe",
        "north america": "North America",
        "na": "North America",
        "south america": "South America",
        "sa": "South America",
        "africa/middle east": "Africa/Middle East",
        "africa": "Africa/Middle East",
        "middle east": "Africa/Middle East",
    }
    canonical_sizes = {
        "large": "Large",
        "medium": "Medium",
        "small": "Small",
    }
    canonical_responses = {
        "customer delay": "Customer Delay",
        "alternative supplier": "Alternative Supplier",
        "production reroute": "Production Reroute",
        "inventory buffer": "Inventory Buffer",
        "combined strategy": "Combined Strategy",
    }

    normalized = {}
    for k, v in filters.items():
        if v is None:
            continue
        key = str(k).strip().lower()
        if key == "disruption_type":
            val_str = str(v).strip().lower()
            normalized["disruption_type"] = canonical_disruptions.get(val_str, str(v).title())
        elif key == "industry":
            val_str = str(v).strip().lower()
            normalized["industry"] = canonical_industries.get(val_str, str(v).title())
        elif key == "supplier_region":
            val_str = str(v).strip().lower()
            normalized["supplier_region"] = canonical_regions.get(val_str, str(v).title())
        elif key == "supplier_size":
            val_str = str(v).strip().lower()
            normalized["supplier_size"] = canonical_sizes.get(val_str, str(v).title())
        elif key == "response_type":
            val_str = str(v).strip().lower()
            normalized["response_type"] = canonical_responses.get(val_str, str(v).title())
        elif key == "supplier_tier":
            val_str = str(v).strip().lower().replace("tier", "").strip()
            if val_str in {"1", "2", "3", "4"}:
                normalized["supplier_tier"] = val_str
            else:
                normalized["supplier_tier"] = str(v)
        elif key == "disruption_severity":
            if isinstance(v, dict):
                sev_dict = {}
                for op, op_val in v.items():
                    try:
                        clean_val = int(str(op_val).split("/")[0].strip())
                        sev_dict[op] = clean_val
                    except (ValueError, TypeError):
                        pass
                if sev_dict:
                    normalized["disruption_severity"] = sev_dict
            else:
                try:
                    clean_val = int(str(v).split("/")[0].strip())
                    normalized["disruption_severity"] = clean_val
                except (ValueError, TypeError):
                    pass
        elif key == "has_backup_supplier":
            if isinstance(v, bool):
                normalized["has_backup_supplier"] = v
            else:
                normalized["has_backup_supplier"] = str(v).strip().lower() in {"true", "1", "yes"}
        else:
            if k in FILTER_FIELDS:
                normalized[k] = v

    return normalized


class HybridRetriever:
    """Read-only runtime search over prebuilt Chroma and BM25 artifacts."""

    def __init__(self):
        started = time.perf_counter()
        logger.info("Retrieval init: importing Chroma and SentenceTransformer")
        import chromadb
        from sentence_transformers import SentenceTransformer

        logger.info("Retrieval init: opening Chroma index")
        self.client = chromadb.PersistentClient(path=str(settings.path(settings.chroma_path)))
        logger.info("Retrieval init: opening Chroma index finished (%.2fs)", time.perf_counter() - started)
        started = time.perf_counter()
        self.collection = self.client.get_collection(settings.chroma_collection)
        collection_size = self.collection.count()
        logger.info("Retrieval init: Chroma collection ready with %s records (%.2fs)", collection_size, time.perf_counter() - started)
        if collection_size <= 0:
            raise RuntimeError(f"Chroma collection {settings.chroma_collection!r} has no indexed vectors")
        started = time.perf_counter()
        logger.info("Retrieval init: loading embedding model %s", settings.embedding_model)
        self.encoder = SentenceTransformer(settings.embedding_model)
        logger.info("Retrieval init: embedding model ready (%.2fs)", time.perf_counter() - started)
        started = time.perf_counter()
        index_path = settings.path(settings.bm25_index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index missing: {index_path}. Run scripts/build_bm25_index.py")
        artifacts = joblib.load(index_path)
        self.bm25 = artifacts["bm25"]
        self.bm25_ids: list[str] = artifacts["disruption_ids"]
        logger.info("Retrieval init: BM25 index loaded with %s records (%.2fs)", len(self.bm25_ids), time.perf_counter() - started)
        if not self.bm25_ids:
            raise RuntimeError(f"BM25 index is empty: {index_path}")
        if len(self.bm25_ids) != collection_size:
            raise RuntimeError("Chroma and BM25 document counts differ; rebuild both offline indexes")

    @staticmethod
    def _where(filters: dict[str, Any]) -> dict | None:
        clauses = []
        for key, value in (filters or {}).items():
            if key not in FILTER_FIELDS or value is None:
                continue
            if key == "disruption_severity" and isinstance(value, dict):
                for operator in ("$eq", "$gte", "$lte"):
                    if operator in value:
                        clauses.append({key: {operator: int(value[operator])}})
            elif key == "disruption_severity" and isinstance(value, int):
                clauses.append({key: {"$eq": value}})
            else:
                if key == "supplier_tier":
                    value = str(value)
                clauses.append({key: {"$eq": value}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _db_statement(filters: dict[str, Any]):
        statement = select(Disruption)
        for key, value in (filters or {}).items():
            column = FILTER_FIELDS.get(key)
            if column is None or value is None:
                continue
            if key == "disruption_severity" and isinstance(value, dict):
                if "$eq" in value:
                    statement = statement.where(column == int(value["$eq"]))
                if "$gte" in value:
                    statement = statement.where(column >= int(value["$gte"]))
                if "$lte" in value:
                    statement = statement.where(column <= int(value["$lte"]))
            elif key == "disruption_severity" and isinstance(value, int):
                statement = statement.where(column == value)
            else:
                statement = statement.where(column == (str(value) if key == "supplier_tier" else value))
        return statement

    def search(self, db: Session, query: str, filters: dict[str, Any] | None = None, top_k: int = 10) -> list[dict]:
        normalized_filters = normalize_filter_dict(filters)
        pool_size = min(len(self.bm25_ids), max(top_k * 25, 100))
        vector = self.encoder.encode([query], normalize_embeddings=True).astype("float32")[0].tolist()

        # -------------------------------------------------------------
        # 1. Dense Semantic Search (ChromaDB) with progressive fallback
        # -------------------------------------------------------------
        where_clause = self._where(normalized_filters)
        semantic_result = self.collection.query(
            query_embeddings=[vector],
            n_results=pool_size,
            where=where_clause,
            include=["metadatas", "distances"],
        )
        semantic_ids = (semantic_result.get("ids") or [[]])[0]
        semantic_distances = (semantic_result.get("distances") or [[]])[0]
        semantic_metadatas = (semantic_result.get("metadatas") or [[]])[0]

        # Progressive relaxation fallback if strict metadata conjunction yields 0 candidates:
        if not semantic_ids and normalized_filters:
            # Fallback A: Keep primary categorical facets (e.g. industry / region / disruption_type)
            relaxed_filters = {}
            for primary_key in ("disruption_type", "industry", "supplier_region"):
                if primary_key in normalized_filters:
                    relaxed_filters[primary_key] = normalized_filters[primary_key]
            if relaxed_filters and relaxed_filters != normalized_filters:
                logger.info("[CHROMA] Strict filter yielded 0 results; relaxing filters to: %s", relaxed_filters)
                relaxed_where = self._where(relaxed_filters)
                semantic_result = self.collection.query(
                    query_embeddings=[vector],
                    n_results=pool_size,
                    where=relaxed_where,
                    include=["metadatas", "distances"],
                )
                semantic_ids = (semantic_result.get("ids") or [[]])[0]
                semantic_distances = (semantic_result.get("distances") or [[]])[0]
                semantic_metadatas = (semantic_result.get("metadatas") or [[]])[0]

        if not semantic_ids:
            # Fallback B: Unfiltered semantic search (pure vector similarity)
            logger.info("[CHROMA] Relaxed filter yielded 0 results; using unfiltered semantic search fallback")
            semantic_result = self.collection.query(
                query_embeddings=[vector],
                n_results=pool_size,
                include=["metadatas", "distances"],
            )
            semantic_ids = (semantic_result.get("ids") or [[]])[0]
            semantic_distances = (semantic_result.get("distances") or [[]])[0]
            semantic_metadatas = (semantic_result.get("metadatas") or [[]])[0]

        # Chroma uses cosine distance (0 to 2 for normalized vectors).
        # Cosine similarity = 1 - distance.
        semantic_scores = {
            key: max(0.0, min(1.0, 1.0 - float(distance)))
            for key, distance in zip(semantic_ids, semantic_distances)
        }

        # -------------------------------------------------------------
        # 2. Sparse Keyword Search (BM25)
        # -------------------------------------------------------------
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        all_scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)
        keyword_indexes = np.argpartition(all_scores, -pool_size)[-pool_size:] if pool_size < len(all_scores) else np.arange(len(all_scores))
        keyword_indexes = sorted(keyword_indexes.tolist(), key=lambda i: all_scores[i], reverse=True)
        raw_keyword = {self.bm25_ids[i]: float(all_scores[i]) for i in keyword_indexes}

        # -------------------------------------------------------------
        # 3. Logging retrieval details (Requirements 2, 3, 4)
        # -------------------------------------------------------------
        logger.info(
            "\n[CHROMA]\n"
            "Collection: %s\n"
            "Documents count: %d\n"
            "Query: %s\n"
            "Top K: %d\n"
            "Results returned: %d\n"
            "Distances/similarities: %s\n"
            "Metadatas sample: %s",
            settings.chroma_collection,
            self.collection.count() if hasattr(self.collection, "count") else len(semantic_ids),
            query,
            pool_size,
            len(semantic_ids),
            list(zip(semantic_ids[:3], [round(1.0 - float(d), 4) for d in semantic_distances[:3]])),
            semantic_metadatas[:2] if semantic_metadatas else [],
        )

        logger.info(
            "\n[BM25]\n"
            "Index documents: %d\n"
            "Query tokens: %s\n"
            "Results returned: %d\n"
            "Top results: %s\n"
            "Scores: %s",
            len(self.bm25_ids),
            tokens,
            len(raw_keyword),
            list(raw_keyword.items())[:3],
            [round(v, 4) for v in list(raw_keyword.values())[:3]],
        )

        # -------------------------------------------------------------
        # 4. Merge candidates and fetch detailed disruption records
        # -------------------------------------------------------------
        candidate_ids = list(dict.fromkeys(semantic_ids + list(raw_keyword.keys())))
        statement = select(Disruption).where(Disruption.disruption_id.in_(candidate_ids))
        rows = db.scalars(statement).all()

        # Database safety check: auto-ingest if disruptions table was not populated
        if not rows and candidate_ids:
            disruption_count = db.scalar(select(func.count(Disruption.id)))
            if disruption_count == 0:
                logger.warning("Disruption table is empty in database; running automatic ingestion...")
                import sys
                from app.config import PROJECT_ROOT
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from scripts.ingest_dataset import main as run_ingest
                run_ingest()
                rows = db.scalars(statement).all()

        max_keyword = max((max(0.0, value) for value in raw_keyword.values()), default=0.0)
        sem_weight = max(0.0, settings.semantic_weight)
        key_weight = max(0.0, settings.keyword_weight)
        total_weight = sem_weight + key_weight or 1.0

        result = []
        for row in rows:
            semantic_score = semantic_scores.get(row.disruption_id, 0.0)
            keyword_score = max(0.0, raw_keyword.get(row.disruption_id, 0.0)) / max_keyword if max_keyword else 0.0
            hybrid_score = round((sem_weight * semantic_score + key_weight * keyword_score) / total_weight, 4)

            # Log candidate similarity check (Requirement 7)
            logger.debug(
                "[SIMILARITY THRESHOLD]\nCandidate: %s\nSimilarity/distance: %s / %s\nConfigured threshold: None\nAccepted/rejected: Accepted\nReason: Hybrid candidate passed to ranking",
                row.disruption_id, round(semantic_score, 4), round(1.0 - semantic_score, 4)
            )

            result.append({
                "disruption_id": row.disruption_id,
                "semantic_score": round(semantic_score, 4),
                "keyword_score": round(keyword_score, 4),
                "hybrid_score": hybrid_score,
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
            })

        result.sort(key=lambda item: item["hybrid_score"], reverse=True)
        final_results = result[:top_k]

        logger.info(
            "\n[HYBRID RETRIEVAL]\n"
            "Chroma results: %d\n"
            "BM25 results: %d\n"
            "Merged results: %d\n"
            "Final results: %d",
            len(semantic_ids),
            len(raw_keyword),
            len(candidate_ids),
            len(final_results),
        )

        return final_results


def build_chroma_index() -> dict:
    """Explicit offline index rebuild; never called by FastAPI."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from app.database import SessionLocal
    from app.rag.documents import disruption_metadata, disruption_to_document

    chroma_path = settings.path(settings.chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    collection = client.create_collection(settings.chroma_collection, metadata={"hnsw:space": "cosine"})
    model = SentenceTransformer(settings.embedding_model)
    total = 0
    with SessionLocal() as db:
        last_id = 0
        while True:
            rows = db.scalars(select(Disruption).where(Disruption.id > last_id).order_by(Disruption.id).limit(4096)).all()
            if not rows:
                break
            last_id = rows[-1].id
            documents = [disruption_to_document(row) for row in rows]
            embeddings = model.encode(documents, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist()
            collection.add(
                ids=[row.disruption_id for row in rows], documents=documents,
                metadatas=[disruption_metadata(row) for row in rows], embeddings=embeddings,
            )
            total += len(rows)
    return {"collection": settings.chroma_collection, "vector_count": total, "embedding_model": settings.embedding_model}


def build_bm25_index() -> dict:
    """Explicit offline BM25 build; runtime only loads the saved artifact."""
    import re
    from rank_bm25 import BM25Okapi
    from app.database import SessionLocal
    from app.rag.documents import disruption_to_document

    documents: list[str] = []
    ids: list[str] = []
    with SessionLocal() as db:
        last_id = 0
        while True:
            rows = db.scalars(select(Disruption).where(Disruption.id > last_id).order_by(Disruption.id).limit(5000)).all()
            if not rows:
                break
            last_id = rows[-1].id
            documents.extend(disruption_to_document(row) for row in rows)
            ids.extend(row.disruption_id for row in rows)
    if not documents:
        raise RuntimeError("No disruption rows in MySQL. Run scripts/ingest_dataset.py first.")
    corpus = [re.findall(r"[a-z0-9]+", document.lower()) for document in documents]
    index = BM25Okapi(corpus)
    target = settings.path(settings.bm25_index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"bm25": index, "disruption_ids": ids, "document_count": len(ids)}, target, compress=3)
    return {"index_file": str(target), "indexed_documents": len(ids)}
