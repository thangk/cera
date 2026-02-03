# CERA CLI Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY cera/ ./cera/

# Install PyTorch CPU-only (smaller than full GPU build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[ml,search]"

# Set cache directories for HuggingFace models
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers

# Pre-download sentence-transformers model for MAV similarity
# This downloads and caches the model so it's available at runtime without network access
RUN python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(f'Model cached at: {m._modules}')"

# Verify model is cached (will fail build if not)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True); print('Model verified cached')"

# Create directories for volumes
RUN mkdir -p /app/output /app/configs

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run API server
CMD ["python", "-m", "cera", "serve", "--host", "0.0.0.0", "--port", "8000"]
