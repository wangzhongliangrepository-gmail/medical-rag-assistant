"""医疗 RAG 图状态。"""
from typing import TypedDict


class MedState(TypedDict):
    question: str
    use_external: bool               # 用户开关：是否联网查外部源（默认 False=仅内部，快）
    plan_questions: list[str]        # planner 拆出的子问题（保留供展示）
    sub_questions: list[str]         # 子问题队列
    internal_evidence: list[dict]    # 内部源（Qdrant 教材库）证据
    external_evidence: list[dict]    # 外部源（Tavily Web）证据
    evidence: list[dict]             # 融合重排后的证据 [{text, source, score}]
    answer: str
