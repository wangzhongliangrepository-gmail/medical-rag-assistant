"""医疗 RAG 图状态。"""
from typing import TypedDict


class MedState(TypedDict):
    question: str
    evidence: list[dict]   # [{text, source_id, chunk_id, score}, ...] 来自 KB 混合检索
    answer: str
