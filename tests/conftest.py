from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Disruption  # noqa: E402


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def api_app(session_factory):
    def override_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    with session_factory.begin() as db:
        db.add(Disruption(
            disruption_id="DIS-TEST-0001", disruption_type="Port Congestion", industry="Electronics",
            supplier_tier="2", supplier_region="Asia-Pacific", supplier_size="Medium",
            has_backup_supplier=False, disruption_severity=4, production_impact_pct=54.3,
            revenue_loss_usd=11376013.16, response_type="Alternative Supplier", response_time_days=4,
            partial_recovery_days=20, full_recovery_days=61, permanent_supplier_change=False,
        ))
    yield app
    app.dependency_overrides.clear()

