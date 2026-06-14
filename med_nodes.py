"""医疗 RAG 节点（P3：知识融合）。

plan → retrieve_internal(Qdrant 教材) + retrieve_external(Tavily Web)
     → fuse(合并去重重排) → answer(带来源引用)
"""
from pydantic import BaseModel

from config import FUSE_TOP_K, RERANK_TOP_K
from embeddings import rerank
from external import web_search
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
    structured = llm.with_structured_output(SubQuestions, method="json_mode")
    result = structured.invoke([
        ("system", '你是医疗问答规划器，只输出合法 JSON，格式：{"questions": [...]}'),
        ("human", _PLAN_PROMPT.format(question=state["question"])),
    ])
    subs = result.questions or [state["question"]]
    return {"plan_questions": subs, "sub_questions": subs}


# ---------- 内部源：Qdrant 医疗教材库 ----------

def retrieve_internal(state: MedState) -> dict:
    """对每个子问题混合检索教材库，累积去重，标注来源。"""
    seen: set[tuple] = set()
    ev: list[dict] = []
    for sq in state["sub_questions"]:
        for e in search(sq, top_k=RERANK_TOP_K):
            key = (e["source_id"], e["chunk_id"])
            if key not in seen:
                seen.add(key)
                ev.append({"text": e["text"], "source": f"内部·教材#{e['source_id']}"})
    return {"internal_evidence": ev}


# ---------- 外部源：Tavily Web ----------

def retrieve_external(state: MedState) -> dict:
    """对原问题做 Web 搜索（外部源失败则优雅降级为空，不影响内部）。"""
    try:
        hits = web_search(state["question"])
    except Exception as e:
        print(f"[warn] 外部检索失败，仅用内部源：{type(e).__name__}: {e}")
        return {"external_evidence": []}
    # 网页内容常很长，截断到 800 字：过长对重排慢、对作答也无必要
    return {
        "external_evidence": [
            {"text": h["text"][:800], "source": f"外部·{h['title']}（{h['url']}）"}
            for h in hits if h["text"]
        ]
    }


# ---------- 融合：合并内外部 → 统一重排 ----------

def fuse(state: MedState) -> dict:
    """合并内外部证据，对原问题统一重排取 top-k，让最相关的（不论来源）浮上来。"""
    pool = state["internal_evidence"] + state["external_evidence"]
    if not pool:
        return {"evidence": []}
    docs = [e["text"] for e in pool]
    ranked = rerank(state["question"], docs, top_n=min(FUSE_TOP_K, len(docs)))
    return {"evidence": [{**pool[r["index"]], "score": r["relevance_score"]} for r in ranked]}


# ---------- 作答 ----------

_SYSTEM = "你是严谨的医疗知识助手。只依据提供的资料回答，绝不编造；资料不足就如实说明。"

_ANSWER_PROMPT = """根据以下资料回答问题。资料来自内部医学教材与外部网络两类来源。要求：
1. 只用资料中的信息，不要编造。
2. 在关键结论后用 [编号] 标注依据。
3. 若内部与外部资料冲突，指出分歧。
4. 若资料不足以回答，如实说明。

资料：
{context}

问题：{question}
答案："""


def answer(state: MedState, *, llm) -> dict:
    ev = state["evidence"]
    if not ev:
        return {"answer": "根据现有资料无法回答（未检索到相关内容）。"}
    context = "\n\n".join(
        f"[{i + 1}]（{e['source']}）{e['text']}" for i, e in enumerate(ev)
    )
    msg = llm.invoke([
        ("system", _SYSTEM),
        ("human", _ANSWER_PROMPT.format(context=context, question=state["question"])),
    ])
    return {"answer": msg.content.strip()}
