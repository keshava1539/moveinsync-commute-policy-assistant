"""Tests for atomic policy versioning, rollback on failure, and auto-seeding."""
import pytest
from fastapi.testclient import TestClient

from app.api.routes_policy import seed_default_policies
from app.core.policy_store import policy_store
from app.core.vector_store import vector_store
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure clean vector store and policy registry before and after tests."""
    yield
    # We do not wipe permanently so subsequent tests or runs can inspect,
    # but individual tests manage their data state.


def get_token(company_id: str = "apexcorp", role: str = "employee") -> str:
    """Helper to generate JWT tokens for tests."""
    resp = client.post("/api/v1/auth/token", json={
        "company_id": company_id,
        "employee_id": f"TEST-{role.upper()}",
        "role": role
    })
    return resp.json()["access_token"]


def test_startup_auto_seeding():
    """Verify default policies are seeded and active for all 3 fictional companies."""
    seed_default_policies()
    
    for comp in ["apexcorp", "fintechlabs", "healthtech"]:
        token = get_token(company_id=comp, role="employee")
        resp = client.get(
            "/api/v1/policies/active",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["company_id"] == comp
        assert data["active_version"] >= 1
        assert data["total_chunks"] > 0
        assert data["status"] == "active"


def test_policy_ingest_admin_rbac():
    """Verify that only admin users can trigger policy ingestion."""
    company = "rbaccorp"
    employee_token = get_token(company, "employee")
    admin_token = get_token(company, "admin")

    payload = {
        "company_id": company,
        "document_text": "# RBAC Corp\n\n## 1. Test\nTest content.",
        "document_name": "RBAC Test"
    }

    # Employee must be rejected with 403 Forbidden
    resp_emp = client.post(
        "/api/v1/policies/ingest",
        json=payload,
        headers={"Authorization": f"Bearer {employee_token}"}
    )
    assert resp_emp.status_code == 403
    assert "Admin privileges required" in resp_emp.json()["detail"]

    # Admin should succeed
    resp_admin = client.post(
        "/api/v1/policies/ingest",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert resp_admin.status_code == 201
    assert resp_admin.json()["status"] == "active"


def test_v1_to_v2_atomic_pointer_swap():
    """Verify v1 -> v2 pointer swap: once v2 is live, old v1 rules stop returning."""
    company = "swapcorp"
    admin_token = get_token(company, "admin")
    emp_token = get_token(company, "employee")

    # 1. Ingest clean v1
    doc_v1 = """# SwapCorp Policy v1

## 1. Weekend Transport
Weekend cabs are strictly prohibited. Employees must use personal vehicles and submit claims.
"""
    resp_v1 = client.post(
        "/api/v1/policies/ingest",
        json={"company_id": company, "document_text": doc_v1, "document_name": "SwapCorp v1"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    v1_number = resp_v1.json()["active_version"]

    # Query v1 - must find "strictly prohibited"
    hits_v1 = vector_store.query_policy(company, "weekend transport", top_k=2)
    assert any("strictly prohibited" in h["document"] for h in hits_v1)

    # 2. Ingest v2 with contradictory rule
    doc_v2 = """# SwapCorp Policy v2

## 1. Weekend Transport
Weekend cabs are now fully permitted and company-sponsored for all engineering departments without special approval.
"""
    resp_v2 = client.post(
        "/api/v1/policies/ingest",
        json={"company_id": company, "document_text": doc_v2, "document_name": "SwapCorp v2"},
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    v2_number = resp_v2.json()["active_version"]
    assert v2_number > v1_number

    # 3. Query after v2 activation: MUST return "fully permitted" and NOT "strictly prohibited"
    hits_v2 = vector_store.query_policy(company, "weekend transport", top_k=2)
    assert any("fully permitted" in h["document"] for h in hits_v2)
    assert not any("strictly prohibited" in h["document"] for h in hits_v2)

    # Verify active policy endpoint reflects v2
    resp_active = client.get(
        "/api/v1/policies/active",
        headers={"Authorization": f"Bearer {emp_token}"}
    )
    assert resp_active.json()["active_version"] == v2_number


def test_simulated_mid_ingestion_failure_and_rollback():
    """Verify that a mid-flight failure during ingestion rolls back cleanly without corrupting active version."""
    company = "rollbackcorp"

    # Ingest a valid baseline version
    baseline_doc = """# RollbackCorp Standard

## 1. Safety
Standard safety regulations apply.
"""
    resp_base = vector_store.ingest_policy(company, baseline_doc)
    active_base_version = resp_base.active_version

    # Attempt to ingest new multi-section document, but simulate failure at chunk index 2
    failing_doc = """# RollbackCorp vBroken

## 1. Section One
Section one content.

## 2. Section Two
Section two content.

## 3. Section Three
Section three content that will trigger failure.
"""
    with pytest.raises(RuntimeError) as exc_info:
        vector_store.ingest_policy(
            company_id=company,
            markdown_text=failing_doc,
            simulate_failure_at_chunk=2
        )

    assert "Simulated storage/embedding failure" in str(exc_info.value)

    # 1. Verify active version remained unchanged at active_base_version
    current_active = policy_store.get_active_version(company)
    assert current_active == active_base_version

    # 2. Verify partial chunks from failed target version were purged from Chroma
    failed_target_version = active_base_version + 1
    leftover_chunks = vector_store.collection.get(
        where={
            "$and": [
                {"company_id": {"$eq": company}},
                {"policy_version": {"$eq": failed_target_version}}
            ]
        }
    )
    assert len(leftover_chunks["ids"]) == 0

    # 3. Verify clean retry succeeds without duplicates or collision
    retry_resp = vector_store.ingest_policy(
        company_id=company,
        markdown_text=failing_doc,
        simulate_failure_at_chunk=None
    )
    assert retry_resp.active_version == failed_target_version
    assert policy_store.get_active_version(company) == failed_target_version
