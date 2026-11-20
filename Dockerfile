# Multi-stage/Slim Dockerfile for MoveInSync Commute Policy Assistant
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build-essential, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application codebase, policies, tests, and eval benchmark
COPY app/ ./app/
COPY data/ ./data/
COPY tests/ ./tests/
COPY eval/ ./eval/

# Expose FastAPI service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
