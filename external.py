"""外部知识源：Tavily Web 搜索。知识融合的"外部/实时"那一路。

与内部 Qdrant 教材库互补：教材是稳定的基础知识，Web 补时效/最新指南/教材外内容。
"""
from tavily import TavilyClient

from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("请在 .env 设置 TAVILY_API_KEY（在 tavily.com 免费注册获取）")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def web_search(query: str, max_results: int = TAVILY_MAX_RESULTS) -> list[dict]:
    """Tavily 检索，返回 [{text, title, url}]。"""
    resp = _get_client().search(query=query, max_results=max_results)
    return [
        {
            "text": r.get("content", ""),
            "title": r.get("title", ""),
            "url": r.get("url", ""),
        }
        for r in resp.get("results", [])
    ]
