# FastAPI Backend
# 构建参数：
#   INSTALL_GPU_DEPS=true  安装 PyTorch CUDA + vLLM（生产部署，需要 NVIDIA GPU）
#   INSTALL_GPU_DEPS=false 仅安装 CPU 版 PyTorch，跳过 vLLM（CI / 无 GPU 环境）
ARG INSTALL_GPU_DEPS=false

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (gcc for cryptography/bitsandbytes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Re-declare ARG after FROM (ARG before FROM is only for FROM itself)
ARG INSTALL_GPU_DEPS=false

COPY requirements.txt .

# Conditional dependency installation based on GPU availability
# AutoDL 3090 baseline: install PyTorch CUDA 12.8 before vLLM 0.10.2
RUN if [ "$INSTALL_GPU_DEPS" = "true" ]; then \
        echo "Installing GPU dependencies (PyTorch CUDA 12.8 + vLLM 0.10.2)..." && \
        pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
            --index-url https://download.pytorch.org/whl/cu128 && \
        pip install --no-cache-dir vllm==0.10.2; \
    else \
        echo "Installing CPU-only dependencies (skipping vLLM)..." && \
        pip install --no-cache-dir torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    grep -v -E "^(torch|torchvision|torchaudio|vllm)" requirements.txt > requirements-filtered.txt && \
    pip install --no-cache-dir -r requirements-filtered.txt

# Copy application code
COPY . .

# Non-secret runtime defaults. Database and authentication settings are injected by the deploy environment.
# M5 fix: VLLM_ENABLED=false 与 MODEL_PROVIDER=vllm 矛盾——/ready 会因
# MODEL_PROVIDER=vllm 认为 vLLM 必需，但 VLLM_ENABLED=false 表示无 vLLM 服务，
# 导致健康检查永远 503。默认设为空，docker-compose.yml 会正确覆盖两者。
ENV BACKEND_WORKERS=1
ENV REDIS_URL=redis://redis:6379/0
ENV VLLM_ENABLED=false
ENV MODEL_PROVIDER=
ENV VLLM_BASE_URL=http://vllm:8001

EXPOSE 8000

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
