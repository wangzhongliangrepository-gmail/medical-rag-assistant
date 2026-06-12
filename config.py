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
EMBED_MODEL_UID = os.getenv("EMBED_MODEL_UID", "bge-large-zh-v1.5")
RERANK_MODEL_UID = os.getenv("RERANK_MODEL_UID", "bge-reranker-large")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))
