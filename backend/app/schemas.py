import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

DisruptionType = Literal["Factory Incident", "Cyber Attack", "Port Congestion", "Labor Strike", "Geopolitical", "Natural Disaster"]
Industry = Literal["Pharmaceuticals", "Electronics", "Consumer Goods", "Automotive", "Aerospace"]
Region = Literal["Asia-Pacific", "Europe", "South America", "North America", "Africa/Middle East"]
SupplierSize = Literal["Large", "Medium", "Small"]
ResponseType = Literal["Customer Delay", "Alternative Supplier", "Production Reroute", "Inventory Buffer", "Combined Strategy"]


class AnalysisInput(BaseModel):
    description: str = Field(min_length=12, max_length=1200)
    disruption_type: DisruptionType
    industry: Industry
    supplier_tier: Literal["1", "2", "3", "4"]
    supplier_region: Region
    supplier_size: SupplierSize
    has_backup_supplier: bool
    disruption_severity: int = Field(ge=1, le=5)
    production_impact_pct: float | None = Field(default=None, ge=0, le=100)
    revenue_loss_usd: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    response_type: ResponseType | None = None
    response_time_days: int | None = Field(default=None, ge=0, le=5000)

    @field_validator("description")
    @classmethod
    def input_guardrail(cls, value: str) -> str:
        normalized = re.sub(r"\s+", " ", value).strip()
        patterns = (r"ignore (all |any )?(previous|prior|system) instructions", r"reveal (the )?(system prompt|secrets|api key)", r"act as (an? )?(admin|system)")
        if any(re.search(pattern, normalized, re.I) for pattern in patterns):
            raise ValueError("Enter event facts only; instruction-like text is not accepted.")
        return normalized


class RetrievalFilters(BaseModel):
    disruption_type: DisruptionType | None = None
    industry: Industry | None = None
    supplier_tier: Literal["1", "2", "3", "4"] | None = None
    supplier_region: Region | None = None
    supplier_size: SupplierSize | None = None
    has_backup_supplier: bool | None = None
    disruption_severity: int | dict[str, int] | None = None
    response_type: ResponseType | None = None

    @field_validator("disruption_severity")
    @classmethod
    def validate_severity_filter(cls, value):
        if isinstance(value, int):
            if value < 1 or value > 5:
                raise ValueError("severity filter must be between 1 and 5")
            return value
        if isinstance(value, dict):
            if not value or set(value) - {"$eq", "$gte", "$lte"}:
                raise ValueError("severity filters support only $eq, $gte and $lte")
            if any(not isinstance(item, int) or item < 1 or item > 5 for item in value.values()):
                raise ValueError("severity filter values must be integers from 1 to 5")
            if "$gte" in value and "$lte" in value and value["$gte"] > value["$lte"]:
                raise ValueError("severity $gte cannot exceed $lte")
        return value


class SimilarCasesInput(BaseModel):
    query: str = Field(min_length=3, max_length=1200)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    top_k: int = Field(default=10, ge=1, le=50)


class PredictionInput(BaseModel):
    disruption_type: DisruptionType
    industry: Industry
    supplier_tier: Literal["1", "2", "3", "4"]
    supplier_region: Region
    supplier_size: SupplierSize
    has_backup_supplier: bool
    disruption_severity: int = Field(ge=1, le=5)


class FeedbackInput(BaseModel):
    analysis_id: int = Field(gt=0)
    useful: bool
    actual_response: ResponseType | None = None
    actual_recovery_days: float | None = Field(default=None, ge=0, le=5000)
    actual_production_impact_pct: float | None = Field(default=None, ge=0, le=100)
    actual_revenue_loss_usd: float | None = Field(default=None, ge=0, le=1_000_000_000_000)
    comment: str | None = Field(default=None, max_length=1000)


class WhatIfInput(BaseModel):
    analysis_id: int | None = None
    cases: list[dict] | None = None


class LLMAnalysisOutput(BaseModel):
    summary: str = Field(max_length=1200)
    recommended_response: str
    alternative_responses: list[str] = Field(max_length=4)
    reasoning: list[str] = Field(max_length=8)
    evidence: list[dict]
    confidence: float = Field(ge=0, le=1)
    human_review_required: bool


class RegisterInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=256)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Name is required")
        return value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if value.count("@") != 1 or "." not in value.rsplit("@", 1)[1]:
            raise ValueError("Enter a valid email address")
        return value


class LoginInput(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.strip().lower()
