# AeroForesight-EU — API image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps kept minimal; torch CPU wheels install cleanly on slim.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Train once at build so the image ships a registered model + report.
RUN python -m aeroforesight.mlops.pipeline || true

EXPOSE 8000
CMD ["uvicorn", "aeroforesight.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
