# Image for the fraud-detection API and web dashboard.
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 fraud-api      then open http://127.0.0.1:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# XGBoost and LightGBM need the OpenMP runtime, which the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install libraries first so Docker can cache this layer between code changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only what the running service needs: no data, tests or training outputs.
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY models/ models/
COPY outputs/ outputs/

# Do not run as root inside the container.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Hosting platforms set PORT; locally it defaults to 8000.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
