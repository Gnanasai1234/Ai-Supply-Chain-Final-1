from __future__ import annotations

import json
import logging
import time
from threading import Event, Thread
from collections import defaultdict
from datetime import date, datetime, time as day_time, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import case, delete, func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.models import AnalysisHistory, AnalysisRequest, AuthSession, Disruption, EvaluationResult, Feedback, ModelVersion, Prediction, Recommendation, User
from app.schemas import AnalysisInput, FeedbackInput, LoginInput, PredictionInput, RegisterInput, SimilarCasesInput, WhatIfInput
from app.security import AuthContext, get_auth_context, hash_password, issue_session, verify_password

app = FastAPI(title="AI Supply Chain Disruption Assistant", version="2.0.0", description="Local supply-chain decision support grounded in historical records and saved ML models.")
logger = logging.getLogger("uvicorn.error")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.exception_handler(SQLAlchemyError)
async def handle_database_error(_request: Request, _exc: SQLAlchemyError) -> JSONResponse:
    """Return a controlled API response for DB failures instead of leaking a traceback."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Database unavailable. Check DATABASE_URL and confirm local MySQL is running."},
    )

retriever = None
retriever_error: str | None = None
retriever_ready = Event()
workflow = None
workflow_error: str | None = None
priority_reference: dict[str, list[float]] = {}
model_availability: dict[str, bool] = {}
database_startup_error: str | None = None


def initialize_retriever() -> None:
    """Load disk-heavy search indexes outside FastAPI's startup critical path."""
    global retriever, retriever_error
    stage_started = time.perf_counter()
    try:
        from app.rag.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        retriever_error = None
        logger.info("Startup: retrieval initialized in background (%.2fs)", time.perf_counter() - stage_started)
    except Exception as exc:
        retriever = None
        retriever_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Startup: retrieval initialization failed (%s)", type(exc).__name__)
    finally:
        retriever_ready.set()


def load_priority_reference(db: Session | None = None) -> dict[str, list[float]]:
    from app.services.analysis import load_priority_reference as _load_reference

    return _load_reference(db)


@app.on_event("startup")
def startup() -> None:
    global retriever, retriever_error, workflow, workflow_error, priority_reference, model_availability, database_startup_error
    logger.info("Startup: initializing database schema")
    stage_started = time.perf_counter()
    try:
        Base.metadata.create_all(engine)
        logger.info("Startup: database schema ready (%.2fs)", time.perf_counter() - stage_started)
        logger.info("Startup: loading priority reference data")
        stage_started = time.perf_counter()
        with SessionLocal() as db:
            priority_reference = load_priority_reference(db)
        database_startup_error = None
        logger.info("Startup: priority reference loaded (%.2fs)", time.perf_counter() - stage_started)
    except Exception as exc:
        priority_reference = load_priority_reference(None)
        database_startup_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Startup: database stage failed (%s), loaded fallback reference", type(exc).__name__)
    retriever = None
    retriever_error = None
    retriever_ready.clear()
    logger.info("Startup: scheduling Chroma, BM25, and embedding model %s in background", settings.embedding_model)
    Thread(target=initialize_retriever, name="retriever-init", daemon=True).start()
    logger.info("Startup: building analysis workflow")
    stage_started = time.perf_counter()
    try:
        from app.agents.workflow import build_workflow
        workflow = build_workflow()
        workflow_error = None
        logger.info("Startup: analysis workflow ready (%.2fs)", time.perf_counter() - stage_started)
    except Exception as exc:
        workflow = None
        workflow_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Startup: workflow initialization failed (%s)", type(exc).__name__)
    logger.info("Startup: loading saved prediction models")
    stage_started = time.perf_counter()
    try:
        from app.ml.predictor import initialize_models
        model_availability = initialize_models()
        logger.info("Startup: prediction models loaded (%.2fs)", time.perf_counter() - stage_started)
    except Exception as exc:
        model_availability = {}
        workflow_error = workflow_error or f"Model initialization failed: {type(exc).__name__}: {exc}"
        logger.warning("Startup: prediction-model initialization failed (%s)", type(exc).__name__)


def require_retriever():
    if retriever is None and not retriever_ready.is_set():
        logger.info("Request is waiting for search indexes and embedding model to finish loading")
        if not retriever_ready.wait(timeout=120):
            logger.error("Search index initialization is still running after 120 seconds")
            raise HTTPException(503, "Hybrid retrieval is still initializing. Please retry shortly.")
    if retriever is None:
        raise HTTPException(503, f"Hybrid retrieval not available: {retriever_error or 'index artifacts are missing'}")
    return retriever


def public_user(user: User) -> dict:
    return {"id": user.id, "name": user.name, "email": user.email, "created_at": user.created_at}


@app.post("/api/auth/register", status_code=201)
def register(payload: RegisterInput, db: Session = Depends(get_db)):
    if db.scalar(select(User.id).where(User.email == payload.email)) is not None:
        raise HTTPException(409, "An account with this email already exists")
    user = User(name=payload.name, email=payload.email, password_hash=hash_password(payload.password), is_active=True)
    db.add(user)
    try:
        db.flush()
        token = issue_session(db, user)
        db.commit()
        db.refresh(user)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with this email already exists") from None
    return {"access_token": token, "token_type": "bearer", "user": public_user(user)}


@app.post("/api/auth/login")
def login(payload: LoginInput, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Email or password is incorrect", headers={"WWW-Authenticate": "Bearer"})
    token = issue_session(db, user)
    db.commit()
    return {"access_token": token, "token_type": "bearer", "user": public_user(user)}


@app.post("/api/auth/logout")
def logout(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    db.execute(delete(AuthSession).where(AuthSession.token_hash == auth.token_hash, AuthSession.user_id == auth.user.id))
    db.commit()
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(auth: AuthContext = Depends(get_auth_context)):
    return public_user(auth.user)


def history_item(row: AnalysisHistory, *, include_response: bool = False) -> dict:
    item = {
        "id": row.id, "query": row.query_text, "created_at": row.created_at,
        "disruption_type": row.disruption_type, "industry": row.industry,
        "supplier_region": row.supplier_region, "supplier_tier": row.supplier_tier,
        "severity": row.severity, "final_provider": row.final_provider,
        "final_model": row.final_model, "fallback_used": row.fallback_used,
        "provider_attempts": row.provider_attempts, "response_time_ms": row.response_time_ms,
        "status": row.status,
    }
    response = row.response_data or {}
    recommendation = response.get("recommendation", {})
    item["summary"] = recommendation.get("summary", "Analysis completed.")
    item["recommended_response"] = recommendation.get("recommended_response")
    if include_response:
        item["request_data"] = row.request_data
        item["response"] = response
    return item


@app.get("/api/history/filters")
def history_filters(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    user_id = auth.user.id
    providers = db.scalars(select(AnalysisHistory.final_provider).where(AnalysisHistory.user_id == user_id).distinct().order_by(AnalysisHistory.final_provider)).all()
    models = db.scalars(select(AnalysisHistory.final_model).where(AnalysisHistory.user_id == user_id).distinct().order_by(AnalysisHistory.final_model)).all()
    def dataset_values(column):
        return db.scalars(select(column).distinct().order_by(column)).all()
    return {
        "providers": providers, "models": models,
        "disruption_types": dataset_values(Disruption.disruption_type),
        "industries": dataset_values(Disruption.industry),
        "supplier_regions": dataset_values(Disruption.supplier_region),
        "severities": [1, 2, 3, 4, 5],
    }


@app.get("/api/history/stats")
def history_stats(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    user_id = auth.user.id
    total = int(db.scalar(select(func.count()).select_from(AnalysisHistory).where(AnalysisHistory.user_id == user_id)) or 0)
    successful = int(db.scalar(select(func.count()).select_from(AnalysisHistory).where(AnalysisHistory.user_id == user_id, AnalysisHistory.status == "success")) or 0)
    fallback = int(db.scalar(select(func.count()).select_from(AnalysisHistory).where(AnalysisHistory.user_id == user_id, AnalysisHistory.fallback_used.is_(True))) or 0)
    average = db.scalar(select(func.avg(AnalysisHistory.response_time_ms)).where(AnalysisHistory.user_id == user_id))
    grouped = db.execute(select(AnalysisHistory.final_provider, func.count()).where(AnalysisHistory.user_id == user_id).group_by(AnalysisHistory.final_provider)).all()
    return {
        "total_analyses": total, "successful_requests": successful, "fallback_requests": fallback,
        "average_response_time_ms": round(float(average), 1) if average is not None else None,
        "provider_counts": {provider: int(count) for provider, count in grouped},
    }


@app.get("/api/history")
def list_history(
    date_preset: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    provider: str | None = None,
    model: str | None = None,
    disruption_type: str | None = None,
    severity: int | None = Query(default=None, ge=1, le=5),
    industry: str | None = None,
    supplier_region: str | None = None,
    search: str | None = Query(default=None, max_length=240),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    conditions = [AnalysisHistory.user_id == auth.user.id]
    now = datetime.now()
    if date_preset:
        if date_preset not in {"today", "last_7_days", "last_30_days", "custom"}:
            raise HTTPException(422, "Unsupported date filter")
        if date_preset == "today":
            conditions.append(AnalysisHistory.created_at >= datetime.combine(date.today(), day_time.min))
        elif date_preset == "last_7_days":
            conditions.append(AnalysisHistory.created_at >= now - timedelta(days=7))
        elif date_preset == "last_30_days":
            conditions.append(AnalysisHistory.created_at >= now - timedelta(days=30))
        elif date_preset == "custom":
            if not date_from or not date_to or date_from > date_to:
                raise HTTPException(422, "Custom date filtering requires a valid date_from and date_to")
    if date_from:
        conditions.append(AnalysisHistory.created_at >= datetime.combine(date_from, day_time.min))
    if date_to:
        conditions.append(AnalysisHistory.created_at < datetime.combine(date_to + timedelta(days=1), day_time.min))
    for column, value in (
        (AnalysisHistory.final_provider, provider), (AnalysisHistory.final_model, model),
        (AnalysisHistory.disruption_type, disruption_type), (AnalysisHistory.severity, severity),
        (AnalysisHistory.industry, industry), (AnalysisHistory.supplier_region, supplier_region),
    ):
        if value is not None:
            conditions.append(column == value)
    if search and search.strip():
        conditions.append(AnalysisHistory.search_text.ilike(f"%{search.strip()}%"))
    base = select(AnalysisHistory).where(*conditions)
    total = int(db.scalar(select(func.count()).select_from(base.subquery())) or 0)
    rows = db.scalars(base.order_by(AnalysisHistory.created_at.desc(), AnalysisHistory.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    pages = (total + page_size - 1) // page_size
    return {"items": [history_item(row) for row in rows], "page": page, "page_size": page_size, "total": total, "total_pages": pages}


@app.get("/api/history/{history_id}")
def get_history(history_id: int, auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    row = db.scalar(select(AnalysisHistory).where(AnalysisHistory.id == history_id, AnalysisHistory.user_id == auth.user.id))
    if row is None:
        raise HTTPException(404, "History item not found")
    return history_item(row, include_response=True)


@app.delete("/api/history/{history_id}")
def delete_history_item(history_id: int, auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    row = db.scalar(select(AnalysisHistory).where(AnalysisHistory.id == history_id, AnalysisHistory.user_id == auth.user.id))
    if row is None:
        raise HTTPException(404, "History item not found")
    analysis_id = row.analysis_request_id
    db.delete(row)
    if analysis_id is not None:
        db.execute(delete(Feedback).where(Feedback.analysis_id == analysis_id))
        db.execute(delete(Recommendation).where(Recommendation.analysis_id == analysis_id))
        db.execute(delete(Prediction).where(Prediction.analysis_id == analysis_id))
        db.execute(delete(AnalysisRequest).where(AnalysisRequest.id == analysis_id))
    db.commit()
    return {"deleted": 1}


@app.delete("/api/history")
def clear_history(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    analysis_ids = db.scalars(select(AnalysisHistory.analysis_request_id).where(
        AnalysisHistory.user_id == auth.user.id, AnalysisHistory.analysis_request_id.is_not(None),
    )).all()
    if analysis_ids:
        db.execute(delete(Feedback).where(Feedback.analysis_id.in_(analysis_ids)))
        db.execute(delete(Recommendation).where(Recommendation.analysis_id.in_(analysis_ids)))
        db.execute(delete(Prediction).where(Prediction.analysis_id.in_(analysis_ids)))
        db.execute(delete(AnalysisRequest).where(AnalysisRequest.id.in_(analysis_ids)))
    result = db.execute(delete(AnalysisHistory).where(AnalysisHistory.user_id == auth.user.id))
    db.commit()
    return {"deleted": int(result.rowcount or 0)}


@app.get("/api/health")
def health():
    try:
        with SessionLocal() as db:
            db.execute(select(1))
        mysql_status = "available"
    except Exception:
        mysql_status = "unavailable"
    from app.ml.predictor import model_errors
    return {
        "status": "ok" if mysql_status == "available" else "unavailable",
        "database": mysql_status,
        "chroma": "available" if retriever is not None else "not available",
        "bm25": "available" if retriever is not None else "not available",
        "langgraph": "available" if workflow is not None else "not available",
        "models": model_availability,
        "llm": settings.llm_provider if settings.llm_provider != "demo" else "LLM not configured",
        "artifact_errors": {"database_startup": database_startup_error, "retrieval": retriever_error, "workflow": workflow_error, "models": model_errors()},
    }


@app.get("/api/disruptions")
def list_disruptions(
    disruption_type: str | None = None,
    industry: str | None = None,
    supplier_region: str | None = None,
    disruption_severity: int | None = Query(default=None, ge=1, le=5),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    statement = select(Disruption)
    for column, value in ((Disruption.disruption_type, disruption_type), (Disruption.industry, industry), (Disruption.supplier_region, supplier_region), (Disruption.disruption_severity, disruption_severity)):
        if value is not None:
            statement = statement.where(column == value)
    count_statement = select(func.count()).select_from(statement.subquery())
    rows = db.scalars(statement.order_by(Disruption.id).offset(offset).limit(limit)).all()
    return {"total": db.scalar(count_statement), "limit": limit, "offset": offset, "results": [disruption_to_api(row) for row in rows]}


def disruption_to_api(row: Disruption) -> dict:
    return {key: getattr(row, key) for key in (
        "id", "disruption_id", "disruption_type", "industry", "supplier_tier", "supplier_region", "supplier_size",
        "has_backup_supplier", "disruption_severity", "production_impact_pct", "revenue_loss_usd", "response_type",
        "response_time_days", "partial_recovery_days", "full_recovery_days", "permanent_supplier_change", "created_at",
    )}


@app.get("/api/disruptions/{disruption_id}")
def disruption_detail(disruption_id: int, db: Session = Depends(get_db)):
    row = db.get(Disruption, disruption_id)
    if row is None:
        raise HTTPException(404, "Disruption record not found")
    return disruption_to_api(row)


@app.post("/api/retrieval/similar")
def similar_cases(payload: SimilarCasesInput, db: Session = Depends(get_db)):
    service = require_retriever()
    filters = payload.filters.model_dump(exclude_none=True)
    result = service.search(db, payload.query, filters, payload.top_k)
    return {"results": result, "count": len(result), "filters": filters, "weights": {"semantic": settings.semantic_weight, "keyword": settings.keyword_weight}}


def model_values(payload) -> dict:
    return {
        "disruption_type": payload.disruption_type, "industry": payload.industry,
        "supplier_tier": str(payload.supplier_tier), "supplier_region": payload.supplier_region,
        "supplier_size": payload.supplier_size, "has_backup_supplier": payload.has_backup_supplier,
        "disruption_severity": payload.disruption_severity,
    }


def persist_prediction(db: Session, values: dict, analysis_id: int | None = None) -> dict:
    from app.ml.predictor import predictions as get_predictions
    result = get_predictions(values)
    if not result["available"]:
        return result
    db.add(Prediction(analysis_id=analysis_id, predictions=result["predictions"], explanations=result["explanations"], model_version=result.get("model_version", "version unavailable")))
    db.flush()
    return result


@app.post("/api/predictions")
def predict(payload: PredictionInput, db: Session = Depends(get_db)):
    result = persist_prediction(db, model_values(payload))
    db.commit()
    if not result["available"]:
        raise HTTPException(503, "Model not trained")
    return {
        "predictions": result["predictions"],
        "explanations": result["explanations"],
        "features_used": result["features_used"],
        "reliability": result.get("reliability", {}),
    }


@app.post("/api/disruptions/analyze")
def analyze_disruption(payload: AnalysisInput, auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    if workflow is None:
        raise HTTPException(503, f"LangGraph workflow unavailable: {workflow_error or 'install the backend requirements'}")
    service = require_retriever()
    values = payload.model_dump()
    analysis = AnalysisRequest(request_data=values)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    ref = priority_reference if (priority_reference and all(len(priority_reference.get(k, [])) > 0 for k in ("disruption_severity", "production_impact_pct", "revenue_loss_usd"))) else load_priority_reference(db)
    state = {
        "user_input": values, "validated_input": values, "db": db, "retriever": service,
        "priority_reference": ref,
    }
    analysis_started = time.perf_counter()
    try:
        result = workflow.invoke(state)
    except (HTTPException, SQLAlchemyError):
        raise
    except Exception as exc:
        logger.exception("Unexpected failure while running disruption analysis")
        raise HTTPException(500, "Analysis failed because of an unexpected server error. Check backend logs for details.") from exc
    prediction_result = result.get("predictions", {})
    if prediction_result.get("available"):
        db.add(Prediction(analysis_id=analysis.id, predictions=prediction_result["predictions"], explanations=prediction_result["explanations"], model_version=prediction_result.get("model_version", "version unavailable")))
    provider_tracking = result.get("provider_tracking") or {
        "final_provider": "Deterministic Fallback", "final_model": "extractive",
        "fallback_used": False, "provider_attempts": [],
    }
    response_time_ms = round((time.perf_counter() - analysis_started) * 1000)
    output = {
        "analysis_id": analysis.id,
        "classification": result.get("classification"),
        "priority": result.get("priority"),
        "similar_cases": result.get("retrieved_cases", []),
        "predictions": prediction_result,
        "response_analysis": result.get("response_analysis", []),
        "recovery_analysis": result.get("recovery_analysis", {}),
        "resilience_analysis": result.get("resilience_analysis", {}),
        "recommendation": result.get("recommendations", {}),
        "evidence": result.get("evidence", []),
        "what_if_simulation": result.get("what_if_simulation", []),
        "explanation": {"impact_feature_importance": result.get("impact_analysis", {}).get("feature_importance", {}), "priority_reasons": result.get("priority", {}).get("priority_reasons", [])},
        "validation": result.get("validation_result", {}),
        "llm_status": result.get("llm_status", "LLM not configured"),
        "provider_tracking": provider_tracking,
    }

    recommendation = output["recommendation"]
    output["recommendation"]["human_review_required"] = recommendation.get("human_review_required", True) or not output["validation"].get("valid", False)

    # -------------------------------------------------------------------------
    # RAGAS Evaluation Strategy (Only Evaluation Method)
    # Evaluates quality but NEVER prevents user from receiving final analysis
    # -------------------------------------------------------------------------
    try:
        from app.evaluation_service import calculate_overall_evaluation, log_evaluation_console
        eval_report = calculate_overall_evaluation(
            input_data=values,
            recommendation=recommendation,
            retrieved_cases=result.get("retrieved_cases", []),
        )
        log_evaluation_console(eval_report, values)
        try:
            db.add(EvaluationResult(
                analysis_id=analysis.id,
                evaluation_type="ragas",
                evaluation_data=eval_report,
            ))
        except Exception:
            pass
    except Exception as eval_exc:
        logger.error("RAGAS evaluation failed: %s", eval_exc)
        eval_report = {
            "overall_score_pct": None,
            "threshold_pct": 75.0,
            "status": "UNAVAILABLE",
            "status_color": "#ef4444",
            "has_warning": True,
            "warnings": [f"RAGAS evaluation failed: {eval_exc}"],
            "evaluation_method": "RAGAS",
            "ragas": {
                "status": "unavailable",
                "overall_score": None,
                "overall_pct": None,
                "metrics": {},
                "error": str(eval_exc),
            },
        }
    output["evaluation"] = eval_report

    analysis.result_data = output
    db.add(Recommendation(analysis_id=analysis.id, recommendation_data=recommendation))
    search_text = " ".join(str(value) for value in (
        payload.description, payload.disruption_type, payload.industry, payload.supplier_region,
        payload.supplier_tier, payload.supplier_size, payload.has_backup_supplier, payload.disruption_severity, recommendation.get("summary", ""),
        recommendation.get("recommended_response", ""),
    ))
    db.add(AnalysisHistory(
        user_id=auth.user.id, analysis_request_id=analysis.id,
        query_text=payload.description, search_text=search_text,
        request_data=values, response_data=output,
        disruption_type=payload.disruption_type, industry=payload.industry,
        supplier_region=payload.supplier_region, supplier_tier=str(payload.supplier_tier),
        severity=payload.disruption_severity,
        final_provider=provider_tracking["final_provider"], final_model=provider_tracking["final_model"],
        fallback_used=bool(provider_tracking["fallback_used"]),
        provider_attempts=provider_tracking["provider_attempts"], response_time_ms=response_time_ms,
        status="success",
    ))
    db.commit()
    return output


@app.post("/api/recommendations")
def get_recommendation(analysis_id: int = Query(gt=0), auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    owned = db.scalar(select(AnalysisHistory.id).where(
        AnalysisHistory.analysis_request_id == analysis_id, AnalysisHistory.user_id == auth.user.id,
    ))
    if owned is None:
        raise HTTPException(404, "Recommendation not found")
    row = db.scalar(select(Recommendation).where(Recommendation.analysis_id == analysis_id).order_by(Recommendation.id.desc()))
    if row is None:
        raise HTTPException(404, "Recommendation not found")
    return row.recommendation_data


@app.post("/api/analysis/what-if")
def simulate_what_if(
    payload: WhatIfInput,
    auth: AuthContext = Depends(get_auth_context),
    db: Session = Depends(get_db),
):
    from app.services.analysis import simulate_what_if_strategies
    cases = payload.cases or []
    req_data = None
    if payload.analysis_id:
        history = db.scalar(
            select(AnalysisHistory).where(
                AnalysisHistory.analysis_request_id == payload.analysis_id,
                AnalysisHistory.user_id == auth.user.id,
            )
        )
        if history and history.response_data:
            cases = history.response_data.get("similar_cases", [])
            req_data = history.request_data
    strategies = simulate_what_if_strategies(cases, req_data)
    return {
        "strategies": strategies,
        "case_count": len(cases),
        "disclaimer": (
            "Historical statistics reflect empirical outcomes observed in matching past events. "
            "Model predictions and historical response outcomes are distinct decision-support metrics."
        ),
    }


@app.get("/api/suppliers/risk")
def supplier_risk_dashboard(
    region: str | None = None,
    tier: str | None = None,
    industry: str | None = None,
    min_severity: int | None = Query(default=None, ge=1, le=5),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    from app.services.supplier_risk import get_supplier_risk_overview
    return get_supplier_risk_overview(
        db=db, region=region, tier=tier, industry=industry, min_severity=min_severity, limit=limit
    )



@app.get("/api/analytics/overview")
def analytics_overview(db: Session = Depends(get_db)):
    row = db.execute(select(
        func.count(Disruption.id), func.avg(Disruption.production_impact_pct), func.sum(Disruption.revenue_loss_usd),
        func.avg(Disruption.response_time_days), func.avg(Disruption.full_recovery_days),
        func.avg(Disruption.has_backup_supplier), func.avg(Disruption.permanent_supplier_change),
    )).one()
    return {
        "total_disruptions": int(row[0] or 0), "average_production_impact_pct": round(float(row[1] or 0), 2),
        "total_revenue_loss_usd": round(float(row[2] or 0), 2), "average_response_time_days": round(float(row[3] or 0), 2),
        "average_recovery_time_days": round(float(row[4] or 0), 2), "backup_supplier_coverage_pct": round(float(row[5] or 0) * 100, 2),
        "permanent_supplier_change_rate_pct": round(float(row[6] or 0) * 100, 2),
    }


@app.get("/api/analytics/disruptions")
def analytics_disruptions(db: Session = Depends(get_db)):
    def counts(column):
        return [{"label": str(label), "count": int(count)} for label, count in db.execute(select(column, func.count()).group_by(column).order_by(func.count().desc())).all()]
    return {
        "by_type": counts(Disruption.disruption_type), "by_industry": counts(Disruption.industry),
        "by_region": counts(Disruption.supplier_region), "by_severity": counts(Disruption.disruption_severity),
    }


@app.get("/api/analytics/responses")
def analytics_responses(db: Session = Depends(get_db)):
    rows = db.execute(select(Disruption.response_type, Disruption.production_impact_pct, Disruption.revenue_loss_usd, Disruption.response_time_days, Disruption.partial_recovery_days, Disruption.full_recovery_days)).all()
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row.response_type].append(row)
    from statistics import median
    return [{
        "response_type": key, "case_count": len(items),
        "median_production_impact_pct": median(row.production_impact_pct for row in items),
        "median_revenue_loss_usd": median(row.revenue_loss_usd for row in items),
        "median_response_time_days": median(row.response_time_days for row in items),
        "median_partial_recovery_days": median(row.partial_recovery_days for row in items),
        "median_full_recovery_days": median(row.full_recovery_days for row in items),
    } for key, items in groups.items()]


@app.get("/api/analytics/recovery")
def analytics_recovery(db: Session = Depends(get_db)):
    bucket = case(
        (Disruption.full_recovery_days <= 14, "0-14 days"),
        (Disruption.full_recovery_days <= 30, "15-30 days"),
        (Disruption.full_recovery_days <= 60, "31-60 days"),
        (Disruption.full_recovery_days <= 90, "61-90 days"),
        (Disruption.full_recovery_days <= 180, "91-180 days"),
        else_="181+ days",
    )
    rows = db.execute(select(bucket.label("recovery_bucket"), func.count()).group_by(bucket).order_by(func.min(Disruption.full_recovery_days))).all()
    from statistics import median
    recovery_values = db.scalars(select(Disruption.full_recovery_days)).all()
    return {"distribution": [{"recovery_bucket": label, "case_count": int(count)} for label, count in rows], "median_full_recovery_days": median(recovery_values) if recovery_values else None}


@app.post("/api/feedback", status_code=201)
def create_feedback(payload: FeedbackInput, auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    owned = db.scalar(select(AnalysisHistory.id).where(
        AnalysisHistory.analysis_request_id == payload.analysis_id, AnalysisHistory.user_id == auth.user.id,
    ))
    if owned is None:
        raise HTTPException(404, "Analysis request not found")
    if db.get(AnalysisRequest, payload.analysis_id) is None:
        raise HTTPException(404, "Analysis request not found")
    row = Feedback(analysis_id=payload.analysis_id, **payload.model_dump(exclude={"analysis_id"}))
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"id": row.id, "message": "Feedback saved"}


def _fetch_user_feedback_records(user_id: int, db: Session) -> list[dict]:
    stmt = (
        select(Feedback, AnalysisHistory)
        .join(AnalysisHistory, Feedback.analysis_id == AnalysisHistory.analysis_request_id)
        .where(AnalysisHistory.user_id == user_id)
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())
    )
    rows = db.execute(stmt).all()
    results = []
    for feedback_row, history_row in rows:
        resp = history_row.response_data or {}
        pred = resp.get("predictions", {}).get("predictions", {}) if isinstance(resp.get("predictions"), dict) else {}
        recs = resp.get("recommendations", {}) if isinstance(resp.get("recommendations"), dict) else {}
        
        pred_rec = pred.get("recovery_time", {}).get("prediction") if isinstance(pred.get("recovery_time"), dict) else None
        pred_imp = pred.get("production_impact", {}).get("prediction") if isinstance(pred.get("production_impact"), dict) else None
        rec_resp = recs.get("recommended_response")

        results.append({
            "id": feedback_row.id,
            "analysis_id": feedback_row.analysis_id,
            "useful": feedback_row.useful,
            "rating": 5.0 if feedback_row.useful else 1.0,
            "actual_response": feedback_row.actual_response,
            "actual_recovery_days": feedback_row.actual_recovery_days,
            "actual_production_impact_pct": feedback_row.actual_production_impact_pct,
            "actual_revenue_loss_usd": feedback_row.actual_revenue_loss_usd,
            "comment": feedback_row.comment,
            "created_at": feedback_row.created_at.isoformat() if feedback_row.created_at else None,
            "disruption_type": history_row.disruption_type,
            "industry": history_row.industry,
            "supplier_region": history_row.supplier_region,
            "supplier_tier": history_row.supplier_tier,
            "severity": history_row.severity,
            "final_provider": history_row.final_provider,
            "recommended_response": rec_resp,
            "predicted_recovery_days": pred_rec,
            "predicted_impact_pct": pred_imp,
        })
    return results


@app.get("/api/feedback")
def get_feedback(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    from app.services.feedback_analytics import analyze_user_feedback
    records = _fetch_user_feedback_records(auth.user.id, db)
    return analyze_user_feedback(records)


@app.get("/api/feedback/stats")
def get_feedback_stats(auth: AuthContext = Depends(get_auth_context), db: Session = Depends(get_db)):
    from app.services.feedback_analytics import analyze_user_feedback
    records = _fetch_user_feedback_records(auth.user.id, db)
    analysis = analyze_user_feedback(records)
    return {
        "stats": analysis["stats"],
        "recurring_issues": analysis["recurring_issues"],
        "improvement_insights": analysis["improvement_insights"],
        "prediction_error_metrics": analysis.get("prediction_error_metrics"),
        "retraining_evaluation": analysis.get("retraining_evaluation"),
        "error_trends": analysis.get("error_trends"),
    }



@app.get("/api/evaluation")
def evaluation(db: Session = Depends(get_db)):
    metrics_path = settings.path(settings.model_metrics_path)
    model_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    records = db.scalars(select(EvaluationResult).where(EvaluationResult.evaluation_type == "ragas").order_by(EvaluationResult.created_at.desc()).limit(50)).all()
    if not records:
        records = db.scalars(select(EvaluationResult).order_by(EvaluationResult.created_at.desc()).limit(50)).all()
    latest_report = None
    report_file = settings.path("evaluation/reports/latest_evaluation.json")
    if report_file.exists():
        try:
            latest_report = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "model_metrics": model_metrics,
        "evaluations": [{"type": row.evaluation_type, "data": row.evaluation_data, "created_at": row.created_at} for row in records],
        "latest_report": latest_report,
        "evaluation_method": "RAGAS",
        "status": "available",
    }


@app.get("/api/evaluation/test-cases")
def evaluation_test_cases():
    return {
        "status": "deprecated",
        "message": "Benchmark evaluation has been removed. RAGAS is the only evaluation method.",
        "evaluation_method": "RAGAS",
    }


@app.get("/api/models")
def model_status():
    path = settings.path(settings.model_metrics_path)
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return {"availability": model_availability, "metrics": metrics, "status": "available" if any(model_availability.values()) else "Model not trained"}
