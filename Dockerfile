# Container for the Streamlit app, for Google Cloud Run, Azure Container Apps or any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# CPU-only PyTorch: the GPU build is several GB and a cloud container has no GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Download the local models at build time, so the first visitor doesn't wait for them.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering as M; \
n='distilbert-base-cased-distilled-squad'; AutoTokenizer.from_pretrained(n); M.from_pretrained(n)"

COPY app.py .
COPY ragbot/ ragbot/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud Run and Azure tell the container which port to use through $PORT.
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/_stcore/health')"
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --client.showErrorDetails=false \
    --server.maxUploadSize=${MAX_UPLOAD_MB:-10} --browser.gatherUsageStats=false
