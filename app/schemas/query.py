from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload submitted by employee to ask a policy question."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The employee's commute question",
        json_schema_extra={"example": "Am I eligible for a cab drop if my shift ends after 8:00 PM?"}
    )


class Citation(BaseModel):
    """Grounded citation linking back to the verified document section."""
    document: str = Field(..., description="Document title")
    section: str = Field(..., description="Section identifier and title")
    chunk_id: str = Field(..., description="Unique chunk ID for trace and audit")


class QueryResponse(BaseModel):
    """Structured response returned to employee."""
    answer: str = Field(..., description="Factual answer or standard desk fallback")
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of cited document sections (maximum 3)"
    )
    company_id: str = Field(..., description="The tenant company verified from JWT")
    policy_version: int = Field(..., description="Active policy version queried")
    status: str = Field(
        ...,
        description="'answered', 'not_covered', or 'system_fallback'"
    )
    latency_ms: float = Field(..., description="Total request latency in milliseconds")
    llm_calls_made: int = Field(..., description="Number of LLM calls made (capped at 2)")
