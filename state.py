from typing import TypedDict


class AgentState(TypedDict):
    question: str
    raw_contexts: list[dict]   # [{title: str, sentences: list[str]}, ...]
    top_contexts: list[str]    # rerank 后 top-k 的格式化文本
    answer: str