from typing import Literal

from pydantic import BaseModel, field_validator

from config import RERANK_TOP_K
from embeddings import rerank
from state import AgentState

_CLASSIFY_PROMPT = """\
判断以下问题的类型：
- comparison：对比两个或多个实体的属性（相同/不同/谁更/是否一样）
- bridge：需要先找到中间事实才能回答，答案藏在不同文章里

问题：{question}"""


class ClassifyResult(BaseModel):
    type: Literal["comparison", "bridge"]


def classify(state: AgentState, *, llm) -> dict:
    structured = llm.with_structured_output(ClassifyResult, method="json_mode")
    result = structured.invoke([
        ("system", '只输出合法 JSON，格式：{"type": "comparison"} 或 {"type": "bridge"}'),
        ("human", _CLASSIFY_PROMPT.format(question=state["question"])),
    ])
    # comparison：直接把原问题作为唯一子问题，跳过 planner
    sub_questions = [state["question"]] if result.type == "comparison" else []
    return {
        "question_type": result.type,
        "sub_questions": sub_questions,
        "evidence": [],
        "revisions": 0,
        "missing_info": "",
    }

_SYSTEM = "你是一个严谨、简洁的问答助手。只根据所给资料回答，不要编造。"

_PLANNER_PROMPT = """\
你是一个多跳问答规划器。把下面的问题拆成 1-3 个子问题，每个子问题都能独立检索。
如果问题本身就是单跳的，只输出一个子问题（即原问题）。

问题：{question}"""

_ANSWER_PROMPT = """\
根据以下检索到的资料，简洁地回答问题。只输出答案本身，不要解释。
用与问题相同的语言回答。

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
    return {"sub_questions": result.questions, "evidence": [], "revisions": 0, "missing_info": ""}


def retrieve(state: AgentState) -> dict:
    sub_q = state["sub_questions"][0]
    remaining = state["sub_questions"][1:]

    docs = [
        c["title"] + " " + " ".join(c["sentences"])
        for c in state["raw_contexts"]
    ]
    results = rerank(sub_q, docs, top_n=RERANK_TOP_K)
    is_comparison = state.get("question_type") == "comparison"
    new_evidence = [
        (
            f"[{i+1}] 标题：{state['raw_contexts'][r['index']]['title']}\n"
            f"内容：{' '.join(state['raw_contexts'][r['index']]['sentences'])}"
            if is_comparison else
            f"[子问题: {sub_q}]\n"
            f"[{i+1}] 标题：{state['raw_contexts'][r['index']]['title']}\n"
            f"内容：{' '.join(state['raw_contexts'][r['index']]['sentences'])}"
        )
        for i, r in enumerate(results)
    ]
    return {
        "sub_questions": remaining,
        "evidence": state["evidence"] + new_evidence,
        "top_contexts": new_evidence,  # M1 兼容
        "last_sub_question": sub_q,
    }


_REFINE_PROMPT = """\
上一个子问题：{last_sub_q}
检索到的证据：
{evidence}

下一个子问题（待精化）：{next_sub_q}

请：
1. 从证据中提取上一个子问题的简短答案（一个短语或名字）。
2. 把这个答案代入下一个子问题，改写成更具体的查询。

输出 JSON：{{"intermediate_answer": "...", "refined_question": "..."}}"""


class RefineResult(BaseModel):
    intermediate_answer: str
    refined_question: str


def refine(state: AgentState, *, llm) -> dict:
    if not state["sub_questions"]:
        return {}

    latest_evidence = "\n\n".join(state["evidence"][-RERANK_TOP_K:])
    structured = llm.with_structured_output(RefineResult, method="json_mode")
    result = structured.invoke([
        ("system", '你是一个信息提取器，只输出合法 JSON，格式：{"intermediate_answer": "...", "refined_question": "..."}'),
        ("human", _REFINE_PROMPT.format(
            last_sub_q=state["last_sub_question"],
            evidence=latest_evidence,
            next_sub_q=state["sub_questions"][0],
        )),
    ])
    return {
        "intermediate_answers": state.get("intermediate_answers", []) + [result.intermediate_answer],
        "sub_questions": [result.refined_question] + state["sub_questions"][1:],
    }


_REFLECT_PROMPT = """\
你是一个严格的质检员。判断当前证据能否支撑答案。

问题：{question}

当前证据：
{evidence}

当前答案：{answer}

判断规则：
- sufficient=true：证据中能找到回答问题所需的关键事实，答案基本正确
- sufficient=false：证据根本不包含答案所需信息，或答案与证据明显矛盾

【重要】不要因为答案可以更完整、更精确就判 false。
只有在证据无法支撑给出可信答案时，才判 sufficient=false。

如果 sufficient=false，missing 必须是一个具体的检索查询，
包含具体实体名、关系或属性，不能是对原问题的复述。
示例：✓ "Shirley Temple 担任的政府职位"  ✗ "那位演员的职位是什么"

输出 JSON：{{"sufficient": true/false, "reasoning": "一句话判断依据", "missing": "（sufficient=false 时必填）具体检索查询"}}"""


class ReflectResult(BaseModel):
    sufficient: bool
    reasoning: str
    missing: str = ""

    @field_validator("missing", mode="before")
    @classmethod
    def coerce_null(cls, v):
        return v or ""


def reflect(state: AgentState, *, llm) -> dict:
    # 只取最近 2*RERANK_TOP_K 段，避免多轮 retry 后 prompt 过长
    evidence_text = "\n\n".join(state["evidence"][-2 * RERANK_TOP_K:]) or "（无）"
    structured = llm.with_structured_output(ReflectResult, method="json_mode")
    result = structured.invoke([
        ("system", '你是一个严格的质检员，只输出合法 JSON，格式：{"sufficient": bool, "reasoning": "...", "missing": "..."}'),
        ("human", _REFLECT_PROMPT.format(
            question=state["question"],
            evidence=evidence_text,
            answer=state["answer"],
        )),
    ])
    return {
        "revisions": state.get("revisions", 0) + 1,
        "reflection": result.reasoning,
        "missing_info": "" if result.sufficient else result.missing,
    }


def prepare_retry(state: AgentState) -> dict:
    """reflect 判断不足后，把 missing_info 推入 sub_questions 供 retrieve 处理。"""
    return {"sub_questions": [state["missing_info"]]}


def answer(state: AgentState, *, llm) -> dict:
    context = "\n\n".join(state["evidence"] or state["top_contexts"])
    prompt = _ANSWER_PROMPT.format(context=context, question=state["question"])
    msg = llm.invoke([("system", _SYSTEM), ("human", prompt)])
    return {"answer": msg.content.strip()}
