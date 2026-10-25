from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import legion_auth, security, sign_in_gates
from app.db.session import get_session
from app.models.user import User
from app.schemas.user import TokenPayload

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)

async def get_current_user(
    db: AsyncSession = Depends(get_session), token: str = Depends(reusable_oauth2)
) -> User:
    try:
        payload = security.decode_access_token(token)
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    result = await db.execute(select(User).where(User.id == token_data.sub))
    user = result.scalars().first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        # Blocked (for a Legion user: the account was blocked in Legion).
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    if sign_in_gates.requires_account() and (
        user.legion_account_id is None or await sign_in_gates.refused(db, user.legion_account_id)
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cloud plan required")
    if await legion_auth.session_revoked(db, user.legion_account_id, payload):
        # Signed out everywhere in Legion after this token was issued.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session ended")
    return user


async def get_current_owner(user: User = Depends(get_current_user)) -> User:
    """The person who runs this self-hosted Hub (internal design notes stage 4.3)."""
    if not user.is_owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the owner of this Hub can do this")
    return user
