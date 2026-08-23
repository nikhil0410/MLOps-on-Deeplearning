# ---------------------------------------------------------------------------
# Inference service image (M2). CPU-only PyTorch keeps the image small.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only torch/torchvision first (from the PyTorch CPU wheel index)
# so the huge default CUDA wheels are never pulled.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2

# Install the remaining pinned runtime dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the trained model artifact.
COPY params.yaml ./params.yaml
COPY src ./src
COPY models ./models

# Run as a non-root user for security.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/models/model.pt \
    LOG_LEVEL=INFO \
    PORT=8000

EXPOSE 8000

# Container-level health check hitting the /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
