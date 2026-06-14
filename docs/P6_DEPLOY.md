# P6 部署：FastAPI + Docker

## 架构

```
┌─ docker-compose ───────────────┐
│  app (FastAPI, :8000)          │  Dockerfile 构建
│  qdrant (向量库服务器, :6333)    │  官方镜像，替代本地落盘模式
└──────────┬─────────────────────┘
           │ host.docker.internal:9997
   ┌───────┴─────────┐
   │ Xinference (GPU) │  留在宿主机（Windows GPU 容器化成本高）
   └─────────────────┘
   DeepSeek / Tavily   外部 API（密钥经 .env 注入）
```

**为什么这样分**：
- **app + qdrant 容器化**：qdrant 服务器版有真正的 HNSW 索引，解决本地模式 >2万点的性能告警。
- **Xinference 留宿主机**：BGE 模型要 GPU，Windows 下 GPU 容器化麻烦，app 通过
  `host.docker.internal` 连它即可。
- **DeepSeek/Tavily** 本就是云端 API，容器内走网络访问，密钥从 `.env` 注入（不烤进镜像）。

## 文件

| 文件 | 作用 |
|------|------|
| `Dockerfile` | app 镜像：装依赖、设 `KMP_DUPLICATE_LIB_OK`、uvicorn 启动 |
| `docker-compose.yml` | 编排 app + qdrant，注入 `QDRANT_URL`/`XINFERENCE_URL` |
| `.dockerignore` | 排除 `.env`/`medical_data`/`qdrant_db` 等 |

## 部署步骤（装好 Docker Desktop 后）

```bash
# 1. 起向量库服务器
docker compose up -d qdrant

# 2. 把医疗 KB 灌进 qdrant 服务器（在宿主机跑，连服务器版）
#    ingest.py 见 QDRANT_URL 即连服务器，且会重切+嵌入(走宿主机 Xinference)
$env:QDRANT_URL="http://localhost:6333"
python ingest.py --recreate

# 3. 构建并启动 app
docker compose up -d --build app

# 4. 访问
#    http://localhost:8000        网页前端
#    http://localhost:8000/docs   FastAPI 自动文档(Swagger)
#    http://localhost:8000/health 健康检查
```

> 前提：宿主机 Xinference 开着（bge-m3 + bge-reranker-v2-m3）。

## 关键环境变量（compose 已注入）

| 变量 | 值 | 说明 |
|------|----|----|
| `QDRANT_URL` | `http://qdrant:6333` | app 连容器内 qdrant；不设则用本地落盘 |
| `XINFERENCE_URL` | `http://host.docker.internal:9997` | app 连宿主机 Xinference |
| `DEEPSEEK_API_KEY` / `TAVILY_API_KEY` | （.env） | 外部 API 密钥 |

代码侧已支持：`vectordb.get_client()` 见 `QDRANT_URL` 即连服务器、否则本地落盘；
`ingest.py --recreate` 同理。**本地开发 → 容器部署只切换环境变量，零代码改动。**

## 云端部署延伸

- app 镜像可推到镜像仓库（如阿里云 ACR），在云主机 `docker compose up`。
- qdrant 可换成 Qdrant Cloud（托管），只改 `QDRANT_URL`。
- Xinference 可部署到带 GPU 的云实例，或换用云端 embedding/rerank 服务。

## 部署实测踩的坑（已修）

实际在 Docker Desktop 上跑通时遇到两个问题，均已修复：

1. **`langchain-community>=1.0` 找不到版本**：该子包最高只到 0.4.2，没跟着 langchain
   升到 1.x。`requirements.txt` 里跨包统一写 `>=1.0` 不可靠（子包版本节奏不同步），
   已改为 `langchain-community>=0.4`。
2. **FastEmbed 的 BM25 模型容器内无法联网下载**（`Could not load model Qdrant/bm25
   from any source`）：fastembed 首次用 BM25 要从 HuggingFace 下载，容器内访问失败。
   解法：把宿主机已缓存的模型（约 90K）打进镜像——`COPY fastembed_cache
   /tmp/fastembed_cache`（`/tmp/fastembed_cache` 是 fastembed 在 Linux 下默认缓存目录），
   运行时零下载。
   - 教训："运行时下载模型"是离线/受限网络部署的常见暗坑，要把模型预置进镜像；
     这与"Xinference 留宿主机"是同一类"外部依赖如何进容器"的权衡。

## 现状

**已在 Docker Desktop 上端到端验证通过**（2026-06）：qdrant 容器灌入 54759 向量，
app 容器经 `host.docker.internal:9997` 连宿主机 Xinference 做 dense 向量化 + 重排、
BM25 走镜像内置缓存，`POST /chat` 返回带 `[编号]` 引用的医疗答案（HTTP 200，约 10s）。
代码已做"本地/服务器双模式"适配，迁移无需改码。
