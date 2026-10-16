FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/opt/cache/tiktoken \
    FASTEMBED_CACHE_PATH=/opt/cache/fastembed

# tesseract for OCR; libpq-dev for psycopg2 (psycopg2-binary bundles libpq, but keep it for safety)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the tokenizer and the open-source CPU embedding model into the image (no downloads at startup).
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
 && python -c "from fastembed import TextEmbedding; list(TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/cache/fastembed').embed(['warm up']))"

COPY app ./app
COPY data ./data
COPY tests ./tests
COPY scripts ./scripts
COPY Makefile pytest.ini ./

# Ensure log dir exists (LOG_FILE=/tmp/app.log is used on Render)
RUN mkdir -p /tmp logs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
