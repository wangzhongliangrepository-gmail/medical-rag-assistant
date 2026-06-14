"""医疗 RAG 节点（P3 知识融合 + P5 记忆）。

短期记忆：contextualize 用会话历史把当前问题改写成自包含问题（指代消解）。
长期记忆：recall_memory 召回用户健康事实注入作答；extract_memory 自动抽取写回。

contextualize → recall_memory → plan → retrieve_internal + retrieve_external
             → fuse → answer → extract_memory
"""
import uuid
from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from config import (
    FUSE_TOP_K,
    HISTORY_WINDOW,
    LLM_FLASH,
    LLM_PRO,
    MEMORY_NAMESPACE,
    MEMORY_RECALL_K,
    RERANK_TOP_K,
)
from embeddings import rerank
from external import web_search
from kb_search import search
from llm import get_llm
from med_state import MedState


def _user_ns(config: RunnableConfig) -> tuple[str, str]:
    """长期记忆按 user_id 隔离的 namespace。"""
    user_id = (config.get("configurable") or {}).get("user_id", "anonymous")
    return (MEMORY_NAMESPACE, user_id)


@lru_cache(maxsize=4)
def _cached_llm(model: str):
    return get_llm(model=model)


def _llm(config: RunnableConfig):
    """按请求参数（model_tier: flash/pro）动态选模型；checkpointer/store 仍共享，切模型不丢记忆。"""
    tier = (config.get("configurable") or {}).get("model_tier", "flash")
    return _cached_llm(LLM_PRO if tier == "pro" else LLM_FLASH)


# ---------- 短期记忆：指代消解 ----------

_CONTEXTUALIZE_PROMPT = """下面是医疗问诊的对话历史和用户最新一句话。
请把最新这句改写成一个**自包含、可独立检索**的问题：把「它/那个/上述/这种药」等指代
替换成历史中明确的实体。若最新这句本身已自包含，原样返回。只输出改写后的问题，不要解释。

对话历史：
{history}

最新这句：{question}

改写后的问题："""


def contextualize(state: MedState, config: RunnableConfig) -> dict:
    """用最近几轮历史把当前问题改写成自包含问题；首轮无历史则原样。"""
    history = state.get("history", [])
    if not history:
        return {"standalone_question": state["question"]}
    llm = _llm(config)
    recent = history[-2 * HISTORY_WINDOW:]  # 一轮含 user+assistant 两条
    history_text = "\n".join(f"{h['role']}：{h['content']}" for h in recent)
    msg = llm.invoke([
        ("system", "你是医疗问诊的指代消解器，只输出改写后的问题本身。"),
        ("human", _CONTEXTUALIZE_PROMPT.format(history=history_text, question=state["question"])),
    ])
    return {"standalone_question": msg.content.strip() or state["question"]}


# ---------- 长期记忆：召回 ----------

def recall_memory(state: MedState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """按 user_id 从长期记忆语义召回与当前问题相关的健康事实。"""
    q = state.get("standalone_question") or state["question"]
    items = store.search(_user_ns(config), query=q, limit=MEMORY_RECALL_K)
    return {"user_memory": [it.value["text"] for it in items]}


# ---------- Planning ----------

_PLAN_PROMPT = """你是医疗问答规划器。把下面的问题拆成 1-3 个可独立检索的子问题，
覆盖问题的不同方面（如疾病机理、药物选择、副作用、用药注意事项等）。
若问题本身很单一，只输出一个子问题（即原问题）。

问题：{question}"""


class SubQuestions(BaseModel):
    questions: list[str]


def plan(state: MedState, config: RunnableConfig) -> dict:
    question = state.get("standalone_question") or state["question"]
    structured = _llm(config).with_structured_output(SubQuestions, method="json_mode")
    result = structured.invoke([
        ("system", '你是医疗问答规划器，只输出合法 JSON，格式：{"questions": [...]}'),
        ("human", _PLAN_PROMPT.format(question=question)),
    ])
    subs = result.questions or [question]
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
    """对原问题做 Web 搜索。用户未开启联网则跳过；失败则优雅降级为空。"""
    if not state.get("use_external"):
        return {"external_evidence": []}
    q = state.get("standalone_question") or state["question"]
    try:
        hits = web_search(q)
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
    q = state.get("standalone_question") or state["question"]
    docs = [e["text"] for e in pool]
    ranked = rerank(q, docs, top_n=min(FUSE_TOP_K, len(docs)))
    return {"evidence": [{**pool[r["index"]], "score": r["relevance_score"]} for r in ranked]}


# ---------- 作答 ----------

_SYSTEM = (
    "你是严谨的医疗知识助手。只依据提供的资料回答，绝不编造；资料不足就如实说明。"
    "若已知用户的健康背景（过敏史/慢病/在用药）与用药建议存在安全冲突，必须主动提示。"
)

_ANSWER_PROMPT = """根据以下资料回答问题。资料来自内部医学教材与外部网络两类来源。要求：
1. 只用资料中的信息，不要编造。
2. 在关键结论后用 [编号] 标注依据。
3. 若内部与外部资料冲突，指出分歧。
4. 若资料不足以回答，如实说明。
5. 结合「用户健康背景」作答；若建议的药物与用户过敏史/禁忌冲突，必须主动提示并给替代方向。
{memory_block}
资料：
{context}

问题：{question}
答案："""


def answer(state: MedState, config: RunnableConfig) -> dict:
    question = state.get("standalone_question") or state["question"]
    llm = _llm(config)
    ev = state["evidence"]
    if not ev:
        ans = "根据现有资料无法回答（未检索到相关内容）。"
        return {"answer": ans, "history": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ans},
        ]}
    context = "\n\n".join(
        f"[{i + 1}]（{e['source']}）{e['text']}" for i, e in enumerate(ev)
    )
    mem = state.get("user_memory") or []
    memory_block = (
        "\n用户健康背景（来自长期记忆，作答务必纳入考虑）：\n"
        + "\n".join(f"- {m}" for m in mem) + "\n"
    ) if mem else ""
    msg = llm.invoke([
        ("system", _SYSTEM),
        ("human", _ANSWER_PROMPT.format(memory_block=memory_block, context=context, question=question)),
    ])
    ans = msg.content.strip()
    return {"answer": ans, "history": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ans},
    ]}


# ---------- 反思：自判证据是否充分（P4 反思回路）----------

_REFLECT_PROMPT = """你是严格的医疗质检员。判断「当前资料」能否支撑对问题的**安全、完整**的回答。

问题：{question}

当前资料：
{context}

当前答案：{answer}

判断规则：
- sufficient=true：资料已覆盖问题的关键方面，答案基本可信、无明显遗漏关键安全信息。
- sufficient=false：资料缺少回答所必需的某个关键方面（如只讲了副作用却没有禁忌/相互作用），
  或缺少与用户健康背景相关的安全信息。

【重要】不要因为「还能更详尽」就判 false；只有缺少**关键**信息时才判 false。
若 sufficient=false，missing 必须是一个**具体的补充检索查询**（含药名/方面），
不能是对原问题的复述。示例：✓「二甲双胍的禁忌症和用药注意」 ✗「还有什么要补充的」

输出 JSON：{{"sufficient": true/false, "reasoning": "一句话理由", "missing": "（不足时必填）具体补充检索查询"}}"""


class ReflectResult(BaseModel):
    sufficient: bool
    reasoning: str
    missing: str = ""


def reflect(state: MedState, config: RunnableConfig) -> dict:
    """答完自判证据是否充分；不足则给出具体补充检索查询。"""
    question = state.get("standalone_question") or state["question"]
    ev = state.get("evidence") or []
    context = "\n\n".join(f"[{i + 1}]（{e['source']}）{e['text']}" for i, e in enumerate(ev)) or "（无）"
    structured = _llm(config).with_structured_output(ReflectResult, method="json_mode")
    try:
        result = structured.invoke([
            ("system", '你是医疗质检员，只输出合法 JSON，格式：{"sufficient": bool, "reasoning": "...", "missing": "..."}'),
            ("human", _REFLECT_PROMPT.format(question=question, context=context, answer=state["answer"])),
        ])
    except Exception as e:
        print(f"[warn] 反思失败，视为充分：{type(e).__name__}: {e}")
        return {"revisions": state.get("revisions", 0) + 1, "reflection": "反思异常", "missing_info": ""}
    return {
        "revisions": state.get("revisions", 0) + 1,
        "reflection": result.reasoning,
        "missing_info": "" if result.sufficient else (result.missing or "").strip(),
    }


def route_after_reflect(state: MedState) -> str:
    """充分或达上限 → 收尾；否则 → 补检索重答。"""
    from config import MAX_REVISIONS
    if not state.get("missing_info"):
        return "extract_memory"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "extract_memory"
    return "augment_retrieve"


def augment_retrieve(state: MedState) -> dict:
    """用反思给出的 missing_info 补检索，结果追加到证据池（去重累积，不覆盖）。"""
    miss = state["missing_info"]

    # 内部库补检索，按 text 去重追加
    internal = list(state.get("internal_evidence") or [])
    seen_int = {e["text"] for e in internal}
    for e in search(miss, top_k=RERANK_TOP_K):
        if e["text"] not in seen_int:
            seen_int.add(e["text"])
            internal.append({"text": e["text"], "source": f"内部·教材#{e['source_id']}"})

    out = {"internal_evidence": internal}

    # 若本轮开了联网，外部也补一路
    if state.get("use_external"):
        external = list(state.get("external_evidence") or [])
        seen_ext = {e["text"] for e in external}
        try:
            for h in web_search(miss):
                t = h["text"][:800]
                if t and t not in seen_ext:
                    seen_ext.add(t)
                    external.append({"text": t, "source": f"外部·{h['title']}（{h['url']}）"})
        except Exception as e:
            print(f"[warn] 反思补检索外部失败，仅补内部：{type(e).__name__}: {e}")
        out["external_evidence"] = external

    return out


# ---------- 长期记忆：自动抽取写回 ----------

_EXTRACT_PROMPT = """从用户这句话里抽取**值得长期记住的健康事实**，只抽取用户**明确陈述**的：
- 过敏史（如对某药/某物过敏）
- 慢性病/既往病史（如高血压、二型糖尿病）
- 长期/正在使用的药物

每条写成一句简洁的事实陈述（如「对青霉素过敏」「患有二型糖尿病」）。
如果这句话里没有这类信息，返回空列表。不要臆测、不要把一次性的症状当成长期事实。

用户这句话：{utterance}"""


class HealthFacts(BaseModel):
    facts: list[str]


def extract_memory(state: MedState, config: RunnableConfig, *, store: BaseStore) -> dict:
    """从本轮用户原始输入抽取健康事实，逐条写入长期记忆。"""
    structured = _llm(config).with_structured_output(HealthFacts, method="json_mode")
    try:
        result = structured.invoke([
            ("system", '你是健康信息抽取器，只输出合法 JSON，格式：{"facts": [...]}'),
            ("human", _EXTRACT_PROMPT.format(utterance=state["question"])),
        ])
    except Exception as e:
        print(f"[warn] 记忆抽取失败，跳过：{type(e).__name__}: {e}")
        return {}
    ns = _user_ns(config)
    for fact in result.facts:
        fact = fact.strip()
        if fact:
            store.put(ns, str(uuid.uuid4()), {"text": fact})
    return {}
