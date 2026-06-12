from typing import TypedDict


class AgentState(TypedDict):
    question: str
    raw_contexts: list[dict]   # [{title: str, sentences: list[str]}, ...]
    top_contexts: list[str]    # rerank 后 top-k 的格式化文本（M1 兼容保留）
    sub_questions: list[str]   # planner 生成的待处理子问题队列
    evidence: list[str]        # 跨子问题累积的格式化段落
    answer: str