"""医疗 RAG 节点：retrieve（查 Qdrant 库）、answer（带引用作答）。"""
from config import RERANK_TOP_K
from kb_search import search
from med_state import MedState

_SYSTEM = "你是严谨的医疗知识助手。只依据提供的资料回答，绝不编造；资料不足就如实说明。"

_ANSWER_PROMPT = """根据以下医学教材资料回答问题。要求：
1. 只用资料中的信息，不要编造或加入资料外的内容。
2. 在关键结论后用 [编号] 标注所依据的资料段。
3. 若资料不足以回答，直接说"根据现有资料无法回答"。

资料：
{context}

问题：{question}
答案："""


def retrieve(state: MedState) -> dict:
    """从 Qdrant 医疗库混合检索 top-k 证据（带来源元数据）。"""
    return {"evidence": search(state["question"], top_k=RERANK_TOP_K)}


def answer(state: MedState, *, llm) -> dict:
    """根据检索证据带引用作答。"""
    ev = state["evidence"]
    if not ev:
        return {"answer": "根据现有资料无法回答（未检索到相关内容）。"}
    context = "\n\n".join(f"[{i + 1}] {e['text']}" for i, e in enumerate(ev))
    msg = llm.invoke([
        ("system", _SYSTEM),
        ("human", _ANSWER_PROMPT.format(context=context, question=state["question"])),
    ])
    return {"answer": msg.content.strip()}
