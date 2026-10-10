from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Disruption(Base):
    __tablename__ = "disruptions"
    __table_args__ = (
        UniqueConstraint("disruption_id", name="uq_disruptions_disruption_id"),
        Index("ix_disruptions_type_industry", "disruption_type", "industry"),
        Index("ix_disruptions_region_tier", "supplier_region", "supplier_tier"),
        Index("ix_disruptions_severity", "disruption_severity"),
        Index("ix_disruptions_response", "response_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    disruption_id: Mapped[str] = mapped_column(String(40), nullable=False)
    disruption_type: Mapped[str] = mapped_column(String(80), nullable=False)
    industry: Mapped[str] = mapped_column(String(80), nullable=False)
    supplier_tier: Mapped[str] = mapped_column(String(30), nullable=False)
    supplier_region: Mapped[str] = mapped_column(String(80), nullable=False)
    supplier_size: Mapped[str] = mapped_column(String(30), nullable=False)
    has_backup_supplier: Mapped[bool] = mapped_column(Boolean, nullable=False)
    disruption_severity: Mapped[int] = mapped_column(Integer, nullable=False)
    production_impact_pct: Mapped[float] = mapped_column(Float, nullable=False)
    revenue_loss_usd: Mapped[float] = mapped_column(Float, nullable=False)
    response_type: Mapped[str] = mapped_column(String(80), nullable=False)
    response_time_days: Mapped[int] = mapped_column(Integer, nullable=False)
    partial_recovery_days: Mapped[int] = mapped_column(Integer, nullable=False)
    full_recovery_days: Mapped[int] = mapped_column(Integer, nullable=False)
    permanent_supplier_change: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class AnalysisRequest(Base):
    __tablename__ = "analysis_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    result_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_requests.id"), nullable=True, index=True)
    predictions: Mapped[dict] = mapped_column(JSON, nullable=False)
    explanations: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False, index=True)
    recommendation_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis_requests.id"), nullable=False, index=True)
    useful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    actual_response: Mapped[str | None] = mapped_column(String(100), nullable=True)
    actual_recovery_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_production_impact_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_revenue_loss_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_type: Mapped[str] = mapped_column(String(80), nullable=False)
    evaluation_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    feature_names: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (Index("ix_auth_sessions_user_expiry", "user_id", "expires_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"
    __table_args__ = (
        Index("ix_analysis_history_user_created", "user_id", "created_at"),
        Index("ix_analysis_history_user_provider", "user_id", "final_provider"),
        Index("ix_analysis_history_user_filters", "user_id", "disruption_type", "industry", "supplier_region", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    analysis_request_id: Mapped[int | None] = mapped_column(ForeignKey("analysis_requests.id", ondelete="SET NULL"), nullable=True, unique=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    search_text: Mapped[str] = mapped_column(Text, nullable=False)
    request_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    response_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    disruption_type: Mapped[str] = mapped_column(String(80), nullable=False)
    industry: Mapped[str] = mapped_column(String(80), nullable=False)
    supplier_region: Mapped[str] = mapped_column(String(80), nullable=False)
    supplier_tier: Mapped[str] = mapped_column(String(30), nullable=False)
    severity: Mapped[int] = mapped_column(Integer, nullable=False)
    final_provider: Mapped[str] = mapped_column(String(80), nullable=False)
    final_model: Mapped[str] = mapped_column(String(160), nullable=False)
    fallback_used: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    provider_attempts: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="success")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
