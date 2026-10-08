# Model berformat JSON, jadi image tidak butuh scikit-learn/statsmodels
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.json

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# modul inferensi: kalender, fitur, dan pembaca model JSON (tanpa sklearn/statsmodels)
COPY src/__init__.py src/calendar_id.py src/features.py src/json_model.py ./src/
COPY app/ ./app/
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
