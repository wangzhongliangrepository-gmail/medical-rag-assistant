"""医疗 RAG 节点：plan（拆子问题）、retrieve（逐子问题查 KB 并汇总）、answer（带引用）。"""
from pydantic import BaseModel

from config import RERANK_TOP_K
from kb_search import search
from med_state import MedState

# ---------- Planning ----------

_PLAN_PROMPT = """你是医疗问答规划器。把下面的问题拆成 1-3 个可独立检索的子问题，
覆盖问题的不同方面（如疾病机理、药物选择、副作用、用药注意事项等）。
若问题本身很单一，只输出一个子问题（即原问题）。

问题：{question}"""


class SubQuestions(BaseModel):
    questions: list[str]


def plan(state: MedState, *, llm) -> dict:
    """把问题拆成覆盖不同方面的子问题。"""
    structured = llm.with_structured_output(SubQuestions, method="json_mode")
    result = structured.invoke([
        ("system", '你是医疗问答规划器，只输出合法 JSON，格式：{"questions": [...]}'),
        ("human", _PLAN_PROMPT.format(question=state["question"])),
    ])
    subs = result.questions or [state["question"]]
    return {"sub_questions": subs}


# ---------- Retrieve ----------

def retrieve(state: MedState) -> dict:
    """对每个子问题混合检索，跨子问题累积证据并按 (source_id, chunk_id) 去重。"""
    seen: set[tuple] = set()
    evidence: list[dict] = []
    for sq in state["sub_questions"]:
        for e in search(sq, top_k=RERANK_TOP_K):
            key = (e["source_id"], e["chunk_id"])
            if key not in seen:
                seen.add(key)
                evidence.append(e)
    return {"evidence": evidence}


# ---------- Answer ----------

_SYSTEM = "你是严谨的医疗知识助手。只依据提供的资料回答，绝不编造；资料不足就如实说明。"

_ANSWER_PROMPT = """根据以下医学教材资料回答问题。要求：
1. 只用资料中的信息，不要编造或加入资料外的内容。
2. 在关键结论后用 [编号] 标注所依据的资料段。
3. 若资料不足以回答（某些方面无依据），就如实说明哪部分无法回答。

资料：
{context}

问题：{question}
答案："""


def answer(state: MedState, *, llm) -> dict:
    """根据累积证据带引用作答。"""
    ev = state["evidence"]
    if not ev:
        return {"answer": "根据现有资料无法回答（未检索到相关内容）。"}
    context = "\n\n".join(f"[{i + 1}] {e['text']}" for i, e in enumerate(ev))
    msg = llm.invoke([
        ("system", _SYSTEM),
        ("human", _ANSWER_PROMPT.format(context=context, question=state["question"])),
    ])
    return {"answer": msg.content.strip()}
