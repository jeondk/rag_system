"""Authentication API endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from supabase import create_client

from langconnect import config
from langconnect.auth import resolve_user

router = APIRouter(prefix="/auth", tags=["auth"])


class SignUpRequest(BaseModel):
    """Sign up request model."""

    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    """Sign in request model."""

    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """Authentication response model."""

    access_token: str
    refresh_token: str
    user_id: str
    email: str


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest) -> AuthResponse:
    """Sign up a new user.

    Args:
        request: Sign up request containing email and password

    Returns:
        AuthResponse with tokens and user information

    Raises:
        HTTPException: If sign up fails
    """
    if config.IS_TESTING:
        raise HTTPException(
            status_code=400,
            detail="Authentication endpoints are disabled in testing mode",
        )

    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    try:
        response = supabase.auth.sign_up(
            {"email": request.email, "password": request.password}
        )

        if not response.session:
            raise HTTPException(
                status_code=400,
                detail="Sign up failed. Please check your email for confirmation.",
            )

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email,
        )

    except Exception as e:
        if "User already registered" in str(e):
            raise HTTPException(status_code=400, detail="User already exists")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest) -> AuthResponse:
    """Sign in an existing user.

    Args:
        request: Sign in request containing email and password

    Returns:
        AuthResponse with tokens and user information

    Raises:
        HTTPException: If sign in fails
    """
    if config.IS_TESTING:
        raise HTTPException(
            status_code=400,
            detail="Authentication endpoints are disabled in testing mode",
        )

    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    try:
        response = supabase.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )

        if not response.session:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email,
        )

    except Exception as e:
        if "Invalid login credentials" in str(e):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/signout")
async def sign_out() -> dict[str, str]:
    """Sign out the current user.

    Note: This endpoint is mainly for client-side cleanup.
    The actual token invalidation happens on the Supabase side.

    Returns:
        Success message
    """
    return {"message": "Successfully signed out"}


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(refresh_token: str) -> AuthResponse:
    """Refresh the access token using a refresh token.

    Args:
        refresh_token: The refresh token

    Returns:
        AuthResponse with new tokens and user information

    Raises:
        HTTPException: If refresh fails
    """
    if config.IS_TESTING:
        raise HTTPException(
            status_code=400,
            detail="Authentication endpoints are disabled in testing mode",
        )

    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    try:
        response = supabase.auth.refresh_session(refresh_token)

        if not response.session:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            user_id=response.user.id,
            email=response.user.email,
        )

    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.get("/me")
async def get_current_user(user: Any = Depends(resolve_user)) -> dict[str, Any]:
    """Get the current authenticated user's information.

    Args:
        user: Current authenticated user

    Returns:
        User information
    """
    return {
        "user_id": user.user_id,
        "email": user.display_name,
        "is_authenticated": user.is_authenticated,
    }
