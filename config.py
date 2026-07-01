"""集中管理端点与模型标识，全部可被 .env 覆盖。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（config.py 所在处），用于拼项目内的离线资源绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent

# --- DeepSeek 云端 LLM ---
# 模型 ID：
#   deepseek-v4-flash —— 快、便宜，支持工具调用与结构化输出（默认，大多数节点用它）
#   deepseek-v4-pro   —— 更强推理/编码/长上下文（planner、reflect 等难节点可选）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 必填
LLM_FLASH = os.getenv("LLM_FLASH", "deepseek-v4-flash")
LLM_PRO   = os.getenv("LLM_PRO",   "deepseek-v4-pro")
LLM_MODEL = os.getenv("LLM_MODEL", LLM_FLASH)  # 兼容旧用法

# --- Xinference / BGE（向量化 + 重排）---
XINFERENCE_URL = os.getenv("XINFERENCE_URL", "http://localhost:9997")
# 这两个是 Xinference 启动模型时返回的 UID（不是模型名），用 `xinference list` 查
EMBED_MODEL_UID = os.getenv("EMBED_MODEL_UID", "bge-m3")
RERANK_MODEL_UID = os.getenv("RERANK_MODEL_UID", "bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # bge-m3 输出维度

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
# BM25 模型缓存目录：默认指向项目内预置的离线缓存（HF hub 布局），
# clone 下来即开箱即用，无需联网下载，本地/容器一致（Docker 里为 /app/fastembed_cache）。
FASTEMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", str(PROJECT_ROOT / "fastembed_cache"))
DENSE_VECTOR_NAME = "dense"     # Qdrant 命名向量：稠密
SPARSE_VECTOR_NAME = "sparse"   # Qdrant 命名向量：稀疏
RECALL_K = int(os.getenv("RECALL_K", "20"))  # 每路召回条数（融合前）

# --- 外部源（Tavily Web 搜索）+ 知识融合 ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
FUSE_TOP_K = int(os.getenv("FUSE_TOP_K", "6"))             # 内外部证据融合重排后保留条数

# --- 文档切块（ingestion）---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# --- 记忆---
# 短期：会话内多轮（checkpointer，按 thread_id）；长期：跨会话用户健康事实（Store，按 user_id）。
MEMORY_NAMESPACE = os.getenv("MEMORY_NAMESPACE", "memories")  # 长期记忆 Store 的 namespace 前缀
MEMORY_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "3"))      # 每轮从长期记忆语义召回条数
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "4"))        # 指代消解参考的最近对话轮数
ENTITY_HINT_K = int(os.getenv("ENTITY_HINT_K", "5"))         # 指代消解时注入的候选实体上限（最近优先）

# --- 反思---
# answer 后自判证据是否充分，不足则补检索重答；硬上限防死循环（医疗克制，最多补检索 1 次）。
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

# --- 安全合规 ---
# 强制免责声明：医疗高风险，对外一律附。在 answer 节点确定性追加进 state["answer"]，
# 保证 CLI / Web / 评测三个输出面都带上（别靠 LLM 自觉，它不可靠）。
DISCLAIMER = os.getenv(
    "DISCLAIMER",
    "⚠️ 本回答仅供学习演示，非医疗建议；如有健康问题请咨询专业医师。",
)
