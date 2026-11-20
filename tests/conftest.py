"""Global pytest configuration and fixtures."""
import pytest
from app.config import settings


@pytest.fixture(scope="session", autouse=True)
def force_mock_llm_for_tests():
    """Ensure pytest always executes using the deterministic mock LLM.
    
    Prevents API quota depletion, network dependencies, or external flakiness
    during test runs, even when .env is configured with real LLM credentials.
    """
    original_provider = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    yield
    settings.LLM_PROVIDER = original_provider
