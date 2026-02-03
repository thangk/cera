# CERA CLI Dockerfile
# GPU Support: Uses CUDA 12.1 for GPU-accelerated MDQA metrics
# To use CPU-only (smaller image): change base image to python:3.11-slim
# and uncomment the CPU-only PyTorch line below
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and system dependencies
# Note: CUDA base image has Python 3.10, we install 3.11 and use python -m pip for consistency
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for Python 3.11 specifically
RUN python -m pip install --upgrade pip

# =============================================================================
# LAYER OPTIMIZATION: Install dependencies BEFORE copying source code
# This ensures PyTorch (~2GB) and other deps are cached unless pyproject.toml changes
# =============================================================================

# Copy ONLY dependency files first (changes rarely)
COPY pyproject.toml README.md ./

# Install PyTorch with CUDA support (GPU-accelerated) - CACHED
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (smaller image, ~1GB vs ~5GB):
# RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install sentence-transformers explicitly (needed for model pre-download)
# This is separate because editable installs with dummy packages don't resolve deps properly
RUN python -m pip install --no-cache-dir sentence-transformers

# Install other Python dependencies (without source, just deps) - CACHED
# Create minimal package structure for pip to read dependencies
RUN mkdir -p cera && touch cera/__init__.py && \
    python -m pip install --no-cache-dir -e ".[ml,search]" && \
    rm -rf cera

# Set cache directories for HuggingFace models
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
ENV TOKENIZERS_PARALLELISM=false

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/sentence-transformers

# Pre-download sentence-transformers model for MAV similarity - CACHED
RUN python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print(f'Model cached at: {m._modules}')"

# Verify model is cached (will fail build if not)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True); print('Model verified cached')"

# =============================================================================
# NOW copy source code (this layer rebuilds on code changes, but deps are cached)
# =============================================================================
COPY cera/ ./cera/

# Re-install in editable mode with actual source (fast, deps already installed)
RUN python -m pip install --no-cache-dir -e ".[ml,search]"

# Create directories for volumes
RUN mkdir -p /app/output /app/configs

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run API server
CMD ["python", "-m", "cera", "serve", "--host", "0.0.0.0", "--port", "8000"]
