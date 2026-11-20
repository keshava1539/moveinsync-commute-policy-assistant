from typing import Optional
from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    """Payload for generating mock authentication tokens."""
    company_id: str = Field(
        ...,
        description="Unique ID of the company, e.g. apexcorp",
        json_schema_extra={"example": "apexcorp"}
    )
    employee_id: str = Field(
        ...,
        description="Unique employee identifier",
        json_schema_extra={"example": "EMP-1024"}
    )
    role: str = Field(
        default="employee",
        description="Role: 'employee' or 'admin'",
        json_schema_extra={"example": "employee"}
    )


class TokenResponse(BaseModel):
    """Authentication token response."""
    access_token: str = Field(..., description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token type, always 'bearer'")
    expires_in: int = Field(..., description="Token validity duration in seconds")
    company_id: str = Field(..., description="Associated company ID")
    employee_id: str = Field(..., description="Associated employee ID")
    role: str = Field(..., description="User role")


class JWTPayload(BaseModel):
    """Decoded JWT payload structure."""
    sub: str = Field(..., description="Subject, maps to employee_id")
    company_id: str = Field(..., description="Strict tenant identifier")
    role: str = Field(default="employee", description="User role: 'employee' or 'admin'")
    exp: Optional[int] = None
    iat: Optional[int] = None
