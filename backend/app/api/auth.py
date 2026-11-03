"""
HR Insight - Authentication Router
Provides environment-configurable demo authentication for HR personnel.
"""
import hmac
import os
import secrets

from fastapi import APIRouter, HTTPException, status
from app.schemas.schemas import LoginRequest, LoginResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

LOCAL_DEMO_USERS = {
    "admin": ("HR Director (Admin)", "Lead HR Analyst", "hr.analytics@enterprise.org"),
    "analyst": ("People Operations Specialist", "HR Analyst", "operations@enterprise.org"),
}


@router.post("/login", response_model=LoginResponse)
def login_for_access_token(credentials: LoginRequest):
    """Authenticate configured HR demo users."""
    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    username = credentials.username.strip().lower()
    password_env = {
        "admin": "HR_INSIGHT_ADMIN_PASSWORD",
        "analyst": "HR_INSIGHT_ANALYST_PASSWORD",
    }.get(username)
    user_info = LOCAL_DEMO_USERS.get(username)
    configured_password = os.getenv(password_env) if password_env else None
    if user_info is None or not configured_password:
        if username == "admin" and is_production:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="HR Insight authentication is not configured.",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(configured_password.encode(), credentials.password.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "access_token": secrets.token_urlsafe(32),
        "token_type": "bearer",
        "user": {
            "username": username,
            "name": user_info[0],
            "role": user_info[1],
            "email": user_info[2],
        }
    }
