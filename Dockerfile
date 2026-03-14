# 1. Upgrade Python for a free speed boost (3.11/3.12 are much faster than 3.9)
FROM python:3.11-slim

# 2. Add Python-specific environment variables for better container behavior
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies and clean up in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# 3. Switch to the user BEFORE installing Python packages
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# 4. Copy requirements with correct ownership in a single step
COPY --chown=appuser:appuser requirements.txt .

# Install dependencies as the non-root user
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# 5. Copy the rest of the application code with ownership in one step
COPY --chown=appuser:appuser . .

EXPOSE 7860

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]