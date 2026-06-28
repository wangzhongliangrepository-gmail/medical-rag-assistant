"""Xinference 上的 BGE 向量化与重排。
  传 server_url + model_uid（UID，不是模型名）。

带退避重试：长时间灌库时 Xinference 偶发 502/网关抖动/连接超时（worker 重启、瞬时拥塞等），
对这类**瞬时**错误自动退避重试，避免整轮灌库/检索因一次抖动白跑。真·宕机（重试到顶仍失败）
照常抛出。
"""
import os
import time

import requests
from langchain_community.embeddings import XinferenceEmbeddings

from config import EMBED_MODEL_UID, RERANK_MODEL_UID, XINFERENCE_URL

# 重试参数（可用环境变量覆盖）；退避 base*2^n：默认 2,4,8,16 秒，最多 4 次
_MAX_RETRIES = int(os.getenv("XINF_MAX_RETRIES", "4"))
_BACKOFF_BASE = float(os.getenv("XINF_RETRY_BASE", "2"))
# 可重试的 HTTP 网关类状态码 + 错误关键词（xinference client 常把 502 包成 RuntimeError）
_RETRY_STATUS = {502, 503, 504, 429}
_RETRY_HINTS = ("502", "503", "504", "bad gateway", "gateway", "timeout",
                "timed out", "connection", "temporarily")


def _is_transient(exc: Exception) -> bool:
    """判断是否「瞬时、可重试」的错误（网关抖动 / 网络 / 超时），而非永久失败。"""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in _RETRY_STATUS
    # 兜底：xinference client 把网关错误包成 RuntimeError，从消息里识别
    msg = str(exc).lower()
    return any(h in msg for h in _RETRY_HINTS)


def _with_retry(fn, *, what: str):
    """对 fn 做退避重试；仅对瞬时错误重试，其余立即抛。"""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  需按错误类型/消息决定是否重试
            if attempt >= _MAX_RETRIES or not _is_transient(e):
                raise
            wait = _BACKOFF_BASE * (2 ** attempt)
            print(f"[retry] {what} 第 {attempt + 1}/{_MAX_RETRIES} 次失败"
                  f"（{type(e).__name__}: {str(e)[:90]}），{wait:.0f}s 后重试…")
            time.sleep(wait)


class RetryXinferenceEmbeddings(XinferenceEmbeddings):
    """XinferenceEmbeddings + 对 502/网络抖动的退避重试（文档批 & 单查询都包）。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _with_retry(
            lambda: XinferenceEmbeddings.embed_documents(self, texts),
            what=f"embed_documents({len(texts)} 段)")

    def embed_query(self, text: str) -> list[float]:
        return _with_retry(
            lambda: XinferenceEmbeddings.embed_query(self, text),
            what="embed_query")


def get_embeddings() -> XinferenceEmbeddings:
    return RetryXinferenceEmbeddings(server_url=XINFERENCE_URL, model_uid=EMBED_MODEL_UID)


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """调用 Xinference /v1/rerank（带退避重试）。

    返回按相关度降序的列表，每项形如：
    {"index": <在 documents 中的下标>, "relevance_score": float, "document": ...}
    """
    payload = {"model": RERANK_MODEL_UID, "query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    def _call():
        resp = requests.post(f"{XINFERENCE_URL}/v1/rerank", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["results"]

    return _with_retry(_call, what="rerank")