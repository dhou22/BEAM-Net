
# BEAM-Net Dockerfile
# ====================
# Base: Python 3.10 (slim) with PyTorch, spiking libraries, Airflow, MLflow
# GPU support via PyTorch CUDA — container detects GPU availability at runtime


FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:/opt/airflow
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=10

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir setuptools
# ---- Stage 1: Install CPU-only PyTorch (smaller, faster) ----
# Official PyTorch CPU index — much smaller than default GPU build
RUN pip install --no-cache-dir \
    --timeout 600 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2 torchvision==0.16.2

# ---- Stage 2: Install all other requirements ----
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
    --timeout 600 --retries 10 \
    -r /tmp/requirements.txt
RUN pip install --no-cache-dir "setuptools==70.0.0"

# Airflow home
ENV AIRFLOW_HOME=/opt/airflow
RUN mkdir -p ${AIRFLOW_HOME}

WORKDIR /workspace