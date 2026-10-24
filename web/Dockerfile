# Image bản server của web SAI: site Astro (bản server) + FastAPI, nạp mô hình từ Hugging Face Hub.
# Build từ gốc repo vì cần SAI/ và hf_release/:  docker compose -f web/compose.yaml build

# --- Stage 1: build site (dữ liệu trong src/data đã được export và commit sẵn) ---
FROM node:22-slim AS site
WORKDIR /src/web/site
ENV ASTRO_TELEMETRY_DISABLED=1
COPY web/site/package.json web/site/package-lock.json ./
RUN npm ci
COPY web/site/ ./
RUN npm run build:server

# --- Stage 2: server Python ---
FROM python:3.13-slim
WORKDIR /app
# torch bản CPU nhỏ hơn nhiều so với bản CUDA mặc định trên PyPI.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.9.1" \
 && pip install --no-cache-dir -r requirements.txt

COPY SAI/ SAI/
COPY hf_release/__init__.py hf_release/checkpoint.py hf_release/
COPY web/__main__.py web/__main__.py
COPY web/backend/ web/backend/
COPY --from=site /src/web/site/dist/server web/site/dist/server

# Dữ liệu người dùng và cache mô hình nằm trong volume (compose.yaml), chạy bằng user thường.
ENV SAI_WEB_DATA_DIR=/data HF_HOME=/cache/huggingface PYTHONUNBUFFERED=1
RUN useradd --create-home sai && mkdir -p /data /cache/huggingface && chown sai /data /cache/huggingface
USER sai

EXPOSE 8000
CMD ["python", "-m", "web", "--source", "hub", "--host", "0.0.0.0", "--port", "8000"]
