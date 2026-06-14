"""医疗 RAG 图状态。"""
from typing import TypedDict


class MedState(TypedDict):
    question: str
    sub_questions: list[str]   # planner 拆出的子问题（覆盖不同方面）
    evidence: list[dict]       # 跨子问题累积去重的证据 [{text, source_id, chunk_id, score}]
    answer: str
