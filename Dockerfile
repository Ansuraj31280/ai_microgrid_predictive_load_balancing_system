# Base image with Python 3.12 (TensorFlow 2.17 supported)
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# Copy app code and artifacts
COPY src ./src
COPY ml_models ./ml_models

# Environment variables for API
ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn worker
CMD ["gunicorn", "src.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
