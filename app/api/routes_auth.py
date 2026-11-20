from datetime import timedelta
from fastapi import APIRouter, Depends, status

from app.api.deps import get_current_user
from app.config import settings
from app.core.security import create_access_token
from app.schemas.auth import JWTPayload, TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Issue Mock JWT Token",
    description="Issues a signed JWT carrying tenant company_id, employee_id, and role."
)
def issue_token(payload: TokenRequest) -> TokenResponse:
    """Issue a mock JWT token for testing and authorization."""
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    claims = {
        "sub": payload.employee_id,
        "company_id": payload.company_id.strip().lower(),
        "role": payload.role.strip().lower()
    }
    token = create_access_token(claims=claims, expires_delta=expires_delta)
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=int(expires_delta.total_seconds()),
        company_id=claims["company_id"],
        employee_id=claims["sub"],
        role=claims["role"]
    )


@router.get(
    "/me",
    response_model=JWTPayload,
    summary="Get Current User Details",
    description="Validates Bearer token and returns the decoded claims."
)
def get_me(current_user: JWTPayload = Depends(get_current_user)) -> JWTPayload:
    """Protected endpoint to inspect the active identity extracted from the JWT."""
    return current_user
