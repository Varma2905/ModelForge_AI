import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token, hash_password, verify_password
from app.database.mongodb import db_client
from app.utils.response import ok

logger = logging.getLogger("regression_studio.auth_routes")

router = APIRouter(prefix="/auth", tags=["Authentication"])


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str


def _to_user_response(user: dict) -> dict:
    return UserResponse(
        id=user["_id"],
        name=user["name"],
        email=user["email"],
        created_at=user["created_at"],
    ).model_dump()


@router.post("/signup")
async def signup(request: SignupRequest):
    normalized_email = request.email.lower()
    existing = await db_client.find_one("users", {"email": normalized_email})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user_doc = {
        "_id": str(ObjectId()),
        "name": request.name.strip(),
        "email": normalized_email,
        "password_hash": hash_password(request.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    user_id = await db_client.insert_one("users", user_doc)
    token = create_access_token(user_id=user_id, email=normalized_email)

    return ok(
        {"user": _to_user_response(user_doc), "token": token},
        message="Account created successfully.",
    )


@router.post("/login")
async def login(request: LoginRequest):
    normalized_email = request.email.lower()
    user = await db_client.find_one("users", {"email": normalized_email})
    if not user or not verify_password(request.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token(
        user_id=user["_id"], email=user["email"], remember_me=request.remember_me
    )
    return ok(
        {"user": _to_user_response(user), "token": token},
        message="Logged in successfully.",
    )


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return ok(_to_user_response(current_user))


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    # Stateless JWT: there is no server-side session to invalidate.
    # The client is responsible for discarding the token.
    return ok(None, message="Logged out successfully.")
