"""Xinference 上的 BGE 向量化与重排。

- 向量化：用 langchain_community 的 XinferenceEmbeddings，需要 `pip install xinference_client`，
  传 server_url + model_uid（UID，不是模型名）。
- 重排：LangChain 没有一等公民的 Xinference 重排器，直接打 /v1/rerank REST 接口最稳，
  避免客户端版本差异。
"""
import requests
from langchain_community.embeddings import XinferenceEmbeddings

from config import EMBED_MODEL_UID, RERANK_MODEL_UID, XINFERENCE_URL


def get_embeddings() -> XinferenceEmbeddings:
    return XinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL_UID)


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """调用 Xinference /v1/rerank。

    返回按相关度降序的列表，每项形如：
    {"index": <在 documents 中的下标>, "relevance_score": float, "document": ...}
    """
    payload = {"model": RERANK_MODEL_UID, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n
    resp = requests.post(f"{XINFERENCE_URL}/v1/rerank", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["results"]
