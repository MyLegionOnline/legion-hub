import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.core.secret_config import configured_secret

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _signing_key() -> str:
    """The key that signs sign-in tokens. There is no built-in default: a known
    key would let anyone mint a token for any user. Without SECRET_KEY the Hub
    signs with a key made at start, so every sign-in ends at the next restart."""
    key = configured_secret("SECRET_KEY")
    if key:
        return key
    logging.getLogger(__name__).warning(
        "SECRET_KEY is not set: signing sign-in tokens with a random key; "
        "everyone is signed out whenever the Hub restarts"
    )
    return secrets.token_urlsafe(48)


SECRET_KEY = _signing_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

def create_access_token(subject: Union[str, Any], expires_delta: timedelta = None,
                        token_version: int | None = None) -> str:
    """``token_version``: the Legion account's ``tv`` at sign-in, so "sign out
    everywhere" in Legion also ends this session (legion_auth.session_revoked)."""
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": str(subject)}
    if token_version is not None:
        to_encode["tv"] = token_version
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> dict:
    """Claims of a sign-in token (:func:`create_access_token`), JWTError otherwise.

    Purpose tokens signed with the same key carry ``typ`` (preview tickets,
    connector tokens, room download links) and some carry a user id in
    ``sub``: they must never pass as a sign-in.
    """
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    if payload.get("typ"):
        raise JWTError("not an access token")
    return payload

def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    # A user created at a Legion sign-in may hold a placeholder instead of a
    # hash (legion_auth.NO_PASSWORD); it simply never matches.
    if not hashed_password:
        return False
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except (ValueError, TypeError):
        return False

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
