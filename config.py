"""集中管理端点与模型标识，全部可被 .env 覆盖。"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- DeepSeek 云端 LLM ---
# 模型 ID：
#   deepseek-v4-flash —— 快、便宜，支持工具调用与结构化输出（默认，大多数节点用它）
#   deepseek-v4-pro   —— 更强推理/编码/长上下文（planner、reflect 等难节点可选）
# 注意：deepseek-chat / deepseek-reasoner 这两个旧别名 2026-07-24 停用，新项目别用。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 必填
LLM_FLASH = os.getenv("LLM_FLASH", "deepseek-v4-flash")
LLM_PRO   = os.getenv("LLM_PRO",   "deepseek-v4-pro")
LLM_MODEL = os.getenv("LLM_MODEL", LLM_FLASH)  # 兼容旧用法

# --- Xinference / BGE（向量化 + 重排，本地，DeepSeek 没有 embedding 接口）---
XINFERENCE_URL = os.getenv("XINFERENCE_URL", "http://localhost:9997")
# 这两个是 Xinference 启动模型时返回的 UID（不是模型名），用 `xinference list` 查
EMBED_MODEL_UID = os.getenv("EMBED_MODEL_UID", "bge-m3")
RERANK_MODEL_UID = os.getenv("RERANK_MODEL_UID", "bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "3"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # bge-large-zh-v1.5 输出维度

# --- Qdrant 向量库（内部知识库）---
# 开发期用本地嵌入模式（落盘，无需 Docker）；部署期改连服务器（QDRANT_URL）。
# 二者只需切换 ingest/检索里的 QdrantClient 构造，集合与数据结构一致。
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")   # 本地落盘目录
QDRANT_URL = os.getenv("QDRANT_URL")                    # 设了则连服务器（如 http://localhost:6333）
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_kb")

# --- 混合检索（dense 语义 + sparse 词面 BM25）---
# dense 来自 bge-m3（Xinference）；sparse 来自 FastEmbed 本地 BM25。
# 两路在 Qdrant 内用 RRF 融合，再过 bge-reranker-v2-m3 重排。
SPARSE_MODEL = os.getenv("SPARSE_MODEL", "Qdrant/bm25")  # FastEmbed 稀疏模型
DENSE_VECTOR_NAME = "dense"     # Qdrant 命名向量：稠密
SPARSE_VECTOR_NAME = "sparse"   # Qdrant 命名向量：稀疏
RECALL_K = int(os.getenv("RECALL_K", "20"))  # 每路召回条数（融合前）

# --- 外部源（Tavily Web 搜索）+ 知识融合 ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")               # 在 tavily.com 注册免费获取
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
FUSE_TOP_K = int(os.getenv("FUSE_TOP_K", "6"))             # 内外部证据融合重排后保留条数

# --- 文档切块（ingestion）---
# BGE-large-zh max_tokens=512，中文约 1 字≈1 token，故 chunk 控制在 400 字以内最稳。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
