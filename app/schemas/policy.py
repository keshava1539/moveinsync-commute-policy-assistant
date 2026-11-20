from typing import Optional
from pydantic import BaseModel, Field


class PolicyIngestRequest(BaseModel):
    """Payload for uploading and ingesting a new policy document."""
    company_id: str = Field(
        ...,
        description="Unique identifier for the company (e.g., apexcorp)",
        json_schema_extra={"example": "apexcorp"}
    )
    document_text: str = Field(
        ...,
        description="Full markdown content of the policy document",
        json_schema_extra={"example": "# ApexCorp Commute Policy\n\n## 1. Eligibility\nAll full-time staff..."}
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Optional human-readable title of the document",
        json_schema_extra={"example": "ApexCorp Commute Policy v2"}
    )


class PolicyIngestResponse(BaseModel):
    """Response returned upon successful atomic ingestion."""
    company_id: str
    active_version: int
    document_name: str
    chunks_ingested: int
    status: str = "active"
    message: str


class PolicyActiveResponse(BaseModel):
    """Details of the currently active policy version for a company."""
    company_id: str
    active_version: int
    document_name: str
    total_chunks: int
    updated_at: str
    status: str = "active"
