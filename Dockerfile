FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System libs required by opencv/paddleocr/vietocr (libgl/libsm/libxext/libxrender),
# torch/paddlepaddle/onnxruntime/faster-whisper (libgomp), audio processing
# (ffmpeg/libsndfile), and building any package without a prebuilt wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch wheels first so the huge CUDA-enabled build from
# PyPI is never pulled (the app always runs with ASR_DEVICE=cpu). The pins in
# requirements.txt below are then already satisfied and pip skips them.
RUN pip install --upgrade pip \
    && pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Default command runs the API. The worker service overrides this in
# docker-compose.yml (same image, different process).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "app"]
