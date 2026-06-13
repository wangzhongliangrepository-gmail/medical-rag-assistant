"""Qdrant 客户端工厂（轻量，仅依赖 qdrant_client + config）。

单独成模块，避免检索路径经 ingest 间接导入 langchain_text_splitters——
后者的原生依赖与 torch/onnxruntime 存在加载顺序冲突，会段错误。
"""
from qdrant_client import QdrantClient

from config import QDRANT_PATH, QDRANT_URL


def get_client() -> QdrantClient:
    """开发期连本地落盘 Qdrant；设了 QDRANT_URL 则连服务器（部署期）。"""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=QDRANT_PATH)
