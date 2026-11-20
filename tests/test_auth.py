"""Unit and integration tests for authentication and health check endpoints."""
from datetime import timedelta
from fastapi.testclient import TestClient
import pytest

from app.core.security import create_access_token
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify service health endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_root_endpoint():
    """Verify root index endpoint responds."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data


def test_issue_token():
    """Verify POST /api/v1/auth/token successfully issues signed JWT with tenant claims."""
    payload = {
        "company_id": "apexcorp",
        "employee_id": "EMP-90210",
        "role": "employee"
    }
    response = client.post("/api/v1/auth/token", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["company_id"] == "apexcorp"
    assert data["employee_id"] == "EMP-90210"
    assert data["role"] == "employee"
    assert data["expires_in"] > 0


def test_auth_me_with_valid_token():
    """Verify GET /api/v1/auth/me accepts valid Bearer token and returns claims."""
    # 1. Get token
    token_resp = client.post("/api/v1/auth/token", json={
        "company_id": "fintechlabs",
        "employee_id": "FT-007",
        "role": "employee"
    })
    token = token_resp.json()["access_token"]

    # 2. Access protected endpoint
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    user_info = response.json()
    assert user_info["company_id"] == "fintechlabs"
    assert user_info["sub"] == "FT-007"
    assert user_info["role"] == "employee"


def test_auth_me_unauthorized_missing_token():
    """Verify accessing protected endpoint without token returns 403 (HTTPBearer)."""
    response = client.get("/api/v1/auth/me")
    assert response.status_code in (401, 403)


def test_auth_me_invalid_token():
    """Verify forged/tampered token returns 401 Unauthorized."""
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer forged-invalid-token-signature"}
    )
    assert response.status_code == 401
    assert "Could not validate credentials" in response.json()["detail"]


def test_auth_me_expired_token():
    """Verify expired token returns 401 with explicit expiration detail."""
    # Create an expired token (negative delta)
    expired_token = create_access_token(
        claims={"sub": "EMP-EXPIRED", "company_id": "apexcorp", "role": "employee"},
        expires_delta=timedelta(seconds=-10)
    )
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()
