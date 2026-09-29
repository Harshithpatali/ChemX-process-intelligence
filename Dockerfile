FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CHEMX_ENV=production \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# The runtime image never receives raw experimental datasets.
COPY src ./src
COPY models ./models
COPY reports ./reports
COPY data/data_dictionary.md ./data/data_dictionary.md
COPY data/README.md ./data/README.md

RUN useradd --create-home --uid 10001 chemx \
    && chown -R chemx:chemx /app

USER chemx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT','8000') + '/health', timeout=3)" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --workers 1"]
