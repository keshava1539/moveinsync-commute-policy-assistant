import os
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_company_id, require_admin
from app.config import settings
from app.core.policy_store import policy_store
from app.core.vector_store import vector_store
from app.schemas.auth import JWTPayload
from app.schemas.policy import (
    PolicyActiveResponse,
    PolicyIngestRequest,
    PolicyIngestResponse,
)

router = APIRouter(prefix="/policies", tags=["Policies"])


@router.post(
    "/ingest",
    response_model=PolicyIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Policy Document (Admin Only)",
    description="Atomically indexes a new policy version for a company. Rolls back safely on failure."
)
def ingest_policy(
    payload: PolicyIngestRequest,
    admin_user: JWTPayload = Depends(require_admin)
) -> PolicyIngestResponse:
    """Ingest a new version of a policy with two-phase pointer swap atomicity."""
    try:
        response = vector_store.ingest_policy(
            company_id=payload.company_id,
            markdown_text=payload.document_text,
            document_name=payload.document_name
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Atomic ingestion failed: {str(exc)}"
        )


@router.get(
    "/active",
    response_model=PolicyActiveResponse,
    summary="Get Active Policy Info",
    description="Retrieves the active policy version and metadata for the authenticated user's company."
)
def get_active_policy(
    company_id: str = Depends(get_company_id)
) -> PolicyActiveResponse:
    """Return currently active policy details for the caller's company."""
    info = policy_store.get_policy_info(company_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active commute policy found for company '{company_id}'."
        )
    return PolicyActiveResponse(**info)


def seed_default_policies() -> None:
    """Auto-seed the 3 default company policies on startup if not already indexed."""
    default_companies = [
        ("apexcorp", "apexcorp_v1.md"),
        ("fintechlabs", "fintechlabs_v1.md"),
        ("healthtech", "healthtech_v1.md")
    ]
    policies_dir = settings.POLICIES_DIR

    for company_id, filename in default_companies:
        active_v = policy_store.get_active_version(company_id)
        if active_v is not None and active_v > 0:
            print(f"[Seed] Company '{company_id}' already has active policy v{active_v}. Skipping seed.")
            continue

        file_path = os.path.join(policies_dir, filename)
        if not os.path.exists(file_path):
            print(f"[Seed] Policy file not found: {file_path}. Skipping.")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            resp = vector_store.ingest_policy(
                company_id=company_id,
                markdown_text=content
            )
            print(f"[Seed] Successfully seeded {company_id}: {resp.message}")
        except Exception as exc:
            print(f"[Seed] Error seeding policy for {company_id}: {exc}")
