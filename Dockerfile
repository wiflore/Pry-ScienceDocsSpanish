FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
RUN pip install --upgrade pip && pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    transformers==4.46.3 \
    torch==2.3.0 \
    pydantic==2.9.2 \
    python-dotenv==1.0.1 \
    google-genai==2.3.0

# Código de la API
COPY api/ ./api/

# Pre-descargar los dos modelos (T1 IMRaD + T2 contribución) 
RUN python -c "\
    from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('wiflore/SciBETO-IMRaD'); \
    AutoModelForSequenceClassification.from_pretrained('wiflore/SciBETO-IMRaD'); \
    AutoTokenizer.from_pretrained('wiflore/SciBETO-T2-contribucion'); \
    AutoModelForSequenceClassification.from_pretrained('wiflore/SciBETO-T2-contribucion')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
