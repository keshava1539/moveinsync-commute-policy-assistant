from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import jwt
from fastapi import HTTPException, status

from app.config import settings
from app.schemas.auth import JWTPayload


def create_access_token(
    claims: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a signed JWT access token encoding the provided claims."""
    to_encode = claims.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp())
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> JWTPayload:
    """Validate and decode a JWT access token into a typed JWTPayload."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        sub: Optional[str] = payload.get("sub")
        company_id: Optional[str] = payload.get("company_id")
        role: str = payload.get("role", "employee")
        exp: Optional[int] = payload.get("exp")
        iat: Optional[int] = payload.get("iat")

        if sub is None or company_id is None:
            raise credentials_exception

        return JWTPayload(
            sub=sub,
            company_id=company_id,
            role=role,
            exp=exp,
            iat=iat
        )

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please request a new token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise credentials_exception
