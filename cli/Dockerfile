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

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create directories for volumes
RUN mkdir -p /app/output /app/configs

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run API server
CMD ["python", "-m", "cera", "serve", "--host", "0.0.0.0", "--port", "8000"]
