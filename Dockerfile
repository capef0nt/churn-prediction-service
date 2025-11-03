# ========= base (build deps cached) =========
FROM python:3.11-slim AS base

# Prevents Python buffering & .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for scientific stack 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# runtime image 
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib

WORKDIR /app

# Create a non-root user
RUN useradd -m appuser
USER appuser

# Copy installed packages from base
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin

# Copy your app code
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser pipeline/ /app/pipeline/
COPY --chown=appuser:appuser tools/ /app/tools/
# Bake the trained artifact INTO the image 
COPY --chown=appuser:appuser artifacts/model.joblib /app/artifacts/model.joblib

# Expose FastAPI port
EXPOSE 8000

# Optional Docker HEALTHCHECK hitting /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python","-c","import json,urllib.request,sys; sys.exit(0 if json.loads(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read().decode()).get('status')=='ok' else 1)"]


# Start server (use a single worker for now)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
