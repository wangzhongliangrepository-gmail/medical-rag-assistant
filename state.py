from typing import TypedDict


class AgentState(TypedDict):
    question: str
    raw_contexts: list[dict]   # [{title: str, sentences: list[str]}, ...]
    top_contexts: list[str]    # rerank 后 top-k 的格式化文本（M1 兼容保留）
    question_type: str         # classify 判断的题型："comparison" 或 "bridge"
    sub_questions: list[str]   # planner 生成的待处理子问题队列
    evidence: list[str]        # 跨子问题累积的格式化段落
    last_sub_question: str     # retrieve 刚处理的子问题（供 refine 使用）
    intermediate_answers: list[str]  # 每跳提取的中间答案
    revisions: int      # 已修订次数，初始 0
    reflection: str     # 最近一次反思理由
    missing_info: str   # 反思指出缺少的信息（空字符串表示 sufficient）
    answer: str