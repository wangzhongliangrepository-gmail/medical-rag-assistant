from pydantic import BaseModel

from config import RERANK_TOP_K
from embeddings import rerank
from state import AgentState

_SYSTEM = "你是一个严谨、简洁的问答助手。只根据所给资料回答，不要编造。"

_PLANNER_PROMPT = """\
你是一个多跳问答规划器。把下面的问题拆成 1-3 个子问题，每个子问题都能独立检索。
如果问题本身就是单跳的，只输出一个子问题（即原问题）。

问题：{question}"""

_ANSWER_PROMPT = """\
根据以下检索到的资料，简洁地回答问题。只输出答案本身，不要解释。

{context}

问题：{question}
答案："""


class SubQuestions(BaseModel):
    questions: list[str]


def planner(state: AgentState, *, llm) -> dict:
    structured = llm.with_structured_output(SubQuestions, method="json_mode")
    result = structured.invoke([
        ("system", '你是一个多跳问答规划器，只输出合法 JSON，格式：{"questions": [...]}'),
        ("human", _PLANNER_PROMPT.format(question=state["question"])),
    ])
    return {"sub_questions": result.questions, "evidence": []}


def retrieve(state: AgentState) -> dict:
    sub_q = state["sub_questions"][0]
    remaining = state["sub_questions"][1:]

    docs = [
        c["title"] + " " + " ".join(c["sentences"])
        for c in state["raw_contexts"]
    ]
    results = rerank(sub_q, docs, top_n=RERANK_TOP_K)
    new_evidence = [
        f"[子问题: {sub_q}]\n"
        f"[{i+1}] 标题：{state['raw_contexts'][r['index']]['title']}\n"
        f"内容：{' '.join(state['raw_contexts'][r['index']]['sentences'])}"
        for i, r in enumerate(results)
    ]
    return {
        "sub_questions": remaining,
        "evidence": state["evidence"] + new_evidence,
        "top_contexts": new_evidence,  # M1 兼容
    }


def answer(state: AgentState, *, llm) -> dict:
    context = "\n\n".join(state["evidence"] or state["top_contexts"])
    prompt = _ANSWER_PROMPT.format(context=context, question=state["question"])
    msg = llm.invoke([("system", _SYSTEM), ("human", prompt)])
    return {"answer": msg.content.strip()}
