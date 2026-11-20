from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token
from app.schemas.auth import JWTPayload

# HTTPBearer extracts "Bearer <token>" from Authorization header
security_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
) -> JWTPayload:
    """Extract and validate the JWT token from the Authorization header."""
    token = credentials.credentials
    return decode_access_token(token)


def get_company_id(
    current_user: JWTPayload = Depends(get_current_user)
) -> str:
    """Strictly resolve the company_id from the verified JWT.
    
    CRITICAL: This value cannot be overridden or modified by any client request payload.
    """
    return current_user.company_id


def require_admin(
    current_user: JWTPayload = Depends(get_current_user)
) -> JWTPayload:
    """Enforce that the requesting user has the 'admin' role."""
    if current_user.role.lower() != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this action."
        )
    return current_user
