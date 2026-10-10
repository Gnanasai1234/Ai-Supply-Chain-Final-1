from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AuthSession, User

SESSION_HOURS = 24
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


@dataclass(frozen=True)
class AuthContext:
    user: User
    token_hash: str


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, salt_hex, digest_hex = encoded.split("$", 2)
        if algorithm != "scrypt":
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def issue_session(db: Session, user: User) -> str:
    """Create a new session token for the user and prune their expired sessions."""
    # Prune expired sessions for this user on every login to keep the table clean.
    db.execute(
        delete(AuthSession).where(
            AuthSession.user_id == user.id,
            AuthSession.expires_at <= datetime.now(timezone.utc).replace(tzinfo=None),
        )
    )
    token = secrets.token_urlsafe(32)
    db.add(AuthSession(
        user_id=user.id,
        token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
        expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=SESSION_HOURS),
    ))
    db.flush()
    return token


def get_auth_context(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> AuthContext:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    auth_session = db.scalar(select(AuthSession).where(
        AuthSession.token_hash == token_hash,
        AuthSession.expires_at > datetime.now(timezone.utc).replace(tzinfo=None),
    ))
    if auth_session is None:
        raise HTTPException(401, "Session is invalid or expired", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(User, auth_session.user_id)
    if user is None or not user.is_active:
        raise HTTPException(401, "Session is invalid or expired", headers={"WWW-Authenticate": "Bearer"})
    return AuthContext(user=user, token_hash=token_hash)
