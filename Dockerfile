# 医疗知识助手 FastAPI 服务镜像
FROM python:3.11-slim

WORKDIR /app

# onnxruntime / fastembed 可能需要的最小编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预置 FastEmbed 的 BM25 模型缓存（容器内无法联网下载 HuggingFace；
# /tmp/fastembed_cache 是 fastembed 在 Linux 下的默认缓存目录）
COPY fastembed_cache /tmp/fastembed_cache

COPY . .

# 放行 torch/onnxruntime 的 OpenMP 重复加载；无缓冲日志
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
