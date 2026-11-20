from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_auth, routes_policy, routes_query
from app.api.routes_policy import seed_default_policies
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle management."""
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # Auto-seed default fictional company policies if not already present
    if settings.AUTO_SEED_ON_STARTUP:
        print("[Startup] Verifying and auto-seeding default company policies...")
        seed_default_policies()
        
    yield
    
    print(f"[{datetime.now(timezone.utc).isoformat()}] Shutting down {settings.PROJECT_NAME}")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Grounded, multi-tenant commute policy assistant with atomic versioning.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routers
    app.include_router(routes_auth.router, prefix=settings.API_V1_PREFIX)
    app.include_router(routes_policy.router, prefix=settings.API_V1_PREFIX)
    app.include_router(routes_query.router, prefix=settings.API_V1_PREFIX)

    @app.get(
        "/health",
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="Service Health Check"
    )
    def health_check() -> Dict[str, str]:
        """Health check endpoint for container probes and monitoring."""
        return {
            "status": "healthy",
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get(
        "/",
        status_code=status.HTTP_200_OK,
        tags=["System"],
        summary="Root Welcome & Index"
    )
    def root() -> Dict[str, str]:
        """Root endpoint directing users to API documentation."""
        return {
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "docs_url": "/docs",
            "health_url": "/health"
        }

    return app


app = create_app()
