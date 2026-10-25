from datetime import timedelta
from logging import getLogger
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import legion_auth, personal_mcp_config, registration, security, sign_in_gates
from app.core.user_provisioning import provision_new_user
from app.core.legion_auth import LegionAuthError
from app.db.session import get_session
from app.models.user import User
from app.schemas.user import UserCreate, UserOut, Token

router = APIRouter()
logger = getLogger(__name__)


async def _token_reply(db: AsyncSession, user: User, **extra) -> dict:
    if sign_in_gates.requires_account():
        if user.legion_account_id is None or await sign_in_gates.refused(db, user.legion_account_id):
            raise _refusal(legion_auth.no_cloud_plan())
    access_token_expires = timedelta(minutes=security.ACCESS_TOKEN_EXPIRE_MINUTES)
    token_version = await legion_auth.current_token_version(db, user.legion_account_id)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires, token_version=token_version,
        ),
        "token_type": "bearer",
        **extra,
    }


def _refusal(exc: LegionAuthError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=exc.detail())


@router.get("/config")
async def auth_config(db: AsyncSession = Depends(get_session)) -> dict:
    """How this Hub signs people in: local passwords, Legion, or both
    (internal design notes §2.2), and whether the fallback form is on.

    On a Hub with local users ``registration`` says who may create an account:
    ``first_user`` (an empty Hub: the next person becomes its owner), ``closed``,
    ``invite`` or ``open`` (internal design notes stage 4.3).
    """
    config = legion_auth.auth_config()
    if not legion_auth.legion_enabled():
        config["registration"] = await registration.effective_mode(db)
    return config


@router.post("/register", response_model=UserOut)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_session)) -> Any:
    if sign_in_gates.requires_account():
        raise HTTPException(status_code=403, detail="Cloud users are created by a paid plan")
    if legion_auth.legion_enabled():
        # Accounts are made in Legion only; while it is down nobody registers.
        if legion_auth.market_available():
            raise HTTPException(status_code=403, detail="Registration goes through your Legion account")
        raise HTTPException(status_code=503, detail="Registration is temporarily unavailable")
    try:
        becomes_owner, invite_id = await registration.admit(
            db, email=user_in.email, invite_token=user_in.invite_token,
        )
    except registration.RegistrationRefused as exc:
        await db.rollback()
        raise HTTPException(status_code=exc.status, detail=exc.detail()) from None
    # Check if user exists
    result = await db.execute(select(User).where(User.email == user_in.email))
    user = result.scalars().first()
    if user:
        await db.rollback()  # an invitation spent on a taken address is given back
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    db_obj = User(
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        full_name=user_in.full_name,
        is_owner=becomes_owner,
    )
    db.add(db_obj)
    await db.flush()
    if invite_id is not None:
        await registration.mark_invite_used_by(db, invite_id, db_obj.id)
    await db.commit()
    await db.refresh(db_obj)
    logger.info(f"User registered successfully: {user_in.email}{' (owner)' if becomes_owner else ''}")

    await provision_new_user(db_obj.id, db)
    if becomes_owner:
        # legion.toml [[personal_mcp]]: installed for the owner, who did not exist before.
        try:
            await personal_mcp_config.apply_declared(db, db_obj)
        except Exception:  # noqa: BLE001 - the account exists; do not fail its registration
            logger.exception("could not install the personal MCP servers declared in legion.toml")

    return db_obj


@router.post("/login", response_model=Token)
async def login(
    request: Request,
    db: AsyncSession = Depends(get_session),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    if legion_auth.legion_enabled():
        return await _legion_password_login(request, form_data, db)
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalars().first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return await _token_reply(db, user)


async def _legion_password_login(request: Request, form_data: OAuth2PasswordRequestForm,
                                 db: AsyncSession) -> dict:
    """The Hub's own password form under AUTH_MODE legion/both.

    ``legion``: only while Market is unreachable (the fallback, plan §2.2);
    otherwise the password is typed in one place, on the Legion page.
    ``both``: always, for the transition. The password is checked against the
    copy Market pushed (legion_credential); a user who has no copy yet (only
    possible in ``both``) falls back to the Hub's own hash.
    """
    if legion_auth.AUTH_MODE == "legion" and legion_auth.market_available():
        raise HTTPException(
            status_code=409,
            detail={"code": "legion_required", "message": "Sign in with your Legion account"},
        )
    email = form_data.username.strip().lower()
    client_ip = request.client.host if request.client else "unknown"
    if not (legion_auth.login_attempt_allowed(f"email:{email}")
            and legion_auth.login_attempt_allowed(f"ip:{client_ip}")):
        raise HTTPException(status_code=429, detail="Too many sign-in attempts, try again later")

    credential = await legion_auth.credential_by_email(db, email)
    if credential is None:
        if legion_auth.AUTH_MODE == "both":
            user = await legion_auth.user_by_email(db, email)
            if user is not None and security.verify_password(form_data.password, user.hashed_password):
                if not user.is_active:
                    raise HTTPException(status_code=400, detail="Inactive user")
                legion_auth.clear_login_attempts(f"email:{email}")
                return await _token_reply(db, user)
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    if not security.verify_password(form_data.password, credential.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    if credential.status != "active":
        raise HTTPException(status_code=403, detail="This Legion account is blocked")
    try:
        # Without Market there is no proof the mailbox is theirs: an existing
        # Hub user with this email is linked only through a Legion sign-in.
        # Nor is a user made here: only a paid plan makes one (A10).
        user, created = await legion_auth.resolve_user(
            db, credential.account_id, credential.email, email_verified=False,
        )
    except LegionAuthError as exc:
        raise _refusal(exc) from None
    if created:  # an invited account's first sign-in (sign-in gates)
        await provision_new_user(user.id, db)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    legion_auth.clear_login_attempts(f"email:{email}")
    logger.info("fallback password login for account %s", credential.account_id)
    return await _token_reply(db, user)


# --- Sign in with Legion (OIDC code + PKCE) -----------------------------------

@router.get("/legion/start")
async def legion_start(intent: Literal["login", "register"] = "login") -> RedirectResponse:
    """Send the browser to the Legion sign-in (or registration) page."""
    if not legion_auth.legion_enabled():
        raise HTTPException(status_code=404, detail="Sign-in with Legion is off on this Hub")
    url, cookie = legion_auth.start_flow(intent)
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        legion_auth.FLOW_COOKIE, cookie, max_age=legion_auth.FLOW_TTL_SEC, path=legion_auth.FLOW_COOKIE_PATH,
        httponly=True, samesite="lax", secure=legion_auth.LEGION_REDIRECT_URI.startswith("https://"),
    )
    return response


class LegionCallback(BaseModel):
    code: str = Field(min_length=1, max_length=512)
    state: str = Field(min_length=1, max_length=512)


@router.post("/legion/callback")
async def legion_callback(body: LegionCallback, request: Request,
                          db: AsyncSession = Depends(get_session)) -> JSONResponse:
    """Finish the sign-in: the SPA posts what Legion put on /auth/legion/callback.

    An account without a Legion Cloud user gets 403 ``no_cloud_plan`` with
    ``pricing_url`` (Market's /pricing?from=hub): a plan payment makes the
    user, a sign-in does not (A10).
    """
    if not legion_auth.legion_enabled():
        raise HTTPException(status_code=404, detail="Sign-in with Legion is off on this Hub")
    try:
        flow = legion_auth.read_flow(request.cookies.get(legion_auth.FLOW_COOKIE), body.state)
        tokens = await legion_auth.exchange_code(body.code, flow["verifier"])
        claims = await legion_auth.verify_id_token(str(tokens.get("id_token", "")), flow["nonce"])
        account_id = UUID(claims["sub"])
        await legion_auth.check_not_revoked(db, account_id, claims["tv"])
        user, created = await legion_auth.resolve_user(
            db, account_id, claims["email"],
            email_verified=bool(claims.get("email_verified")), display_name=claims.get("name"),
        )
    except LegionAuthError as exc:
        reply = JSONResponse({"detail": exc.detail()}, status_code=exc.status)
        reply.delete_cookie(legion_auth.FLOW_COOKIE, path=legion_auth.FLOW_COOKIE_PATH)
        return reply
    if created:  # an invited account's first sign-in (sign-in gates)
        await provision_new_user(user.id, db)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    # A successful exchange also proves Market is up, whatever the last probe said.
    legion_auth.set_market_available(True)
    reply = JSONResponse(await _token_reply(db, user, created=created))

    reply.delete_cookie(legion_auth.FLOW_COOKIE, path=legion_auth.FLOW_COOKIE_PATH)
    return reply
