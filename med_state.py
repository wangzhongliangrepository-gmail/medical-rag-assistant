"""医疗 RAG 图状态。"""
import operator
from typing import Annotated, TypedDict


class MedState(TypedDict):
    question: str                    # 本轮用户原始输入
    use_external: bool               # 用户开关：是否联网查外部源（默认 False=仅内部，快）

    # --- 短期记忆（会话内多轮，靠 checkpointer 跨轮持久化）---
    history: Annotated[list[dict], operator.add]  # [{role, content}]，累积式
    standalone_question: str         # contextualize 用 history 改写后的自包含问题

    # --- 长期用户记忆（本轮召回，临时）---
    user_memory: list[str]           # recall_memory 召回的用户健康事实

    plan_questions: list[str]        # planner 拆出的子问题（保留供展示）
    sub_questions: list[str]         # 子问题队列
    internal_evidence: list[dict]    # 内部源（Qdrant 教材库）证据
    external_evidence: list[dict]    # 外部源（Tavily Web）证据
    evidence: list[dict]             # 融合重排后的证据 [{text, source, score}]
    answer: str

    # --- 反思（P4，反思回路）---
    revisions: int                   # 已反思次数，硬上限 MAX_REVISIONS
    reflection: str                  # 最近一次反思理由
    missing_info: str                # 反思指出缺少的补充检索查询（空=证据充分）
