from functools import partial

from langgraph.graph import END, START, StateGraph

from config import LLM_FLASH, LLM_PRO, MAX_REVISIONS
from llm import get_llm
from nodes import answer, classify, planner, prepare_retry, reflect, refine, retrieve
from state import AgentState


def _route_after_classify(state: AgentState) -> str:
    return "retrieve" if state["question_type"] == "comparison" else "planner"


def _should_continue(state: AgentState) -> str:
    return "retrieve" if state["sub_questions"] else "answer"


def _make_should_reflect_or_end(use_reflect: bool):
    def _should_reflect_or_end(state: AgentState) -> str:
        # comparison 题不走 reflect，直接结束
        if state.get("question_type") == "comparison":
            return "end"
        # ablation：关掉反思回路时，bridge 也直接结束
        return "reflect" if use_reflect else "end"
    return _should_reflect_or_end


def _should_reflect(state: AgentState) -> str:
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "end"
    if not state.get("missing_info"):
        return "end"
    return "retrieve"


def get_graph(tier: str = "flash", use_reflect: bool = True):
    """编译并返回图。

    tier: "flash"（默认）或 "pro"
    use_reflect: True（默认）走完整反思回路；False 为 ablation，bridge 也 answer→END

    comparison 路径：classify → retrieve → refine → answer → END
    bridge 路径：classify → planner → retrieve → refine → answer → reflect → END/retry
    """
    llm_flash = get_llm(model=LLM_FLASH)
    llm_pro   = get_llm(model=LLM_PRO if tier == "pro" else LLM_FLASH)

    g = StateGraph(AgentState)
    g.add_node("classify",     partial(classify, llm=llm_flash))
    g.add_node("planner",      partial(planner,  llm=llm_pro))
    g.add_node("retrieve",     retrieve)
    g.add_node("refine",       partial(refine,   llm=llm_flash))
    g.add_node("answer",       partial(answer,   llm=llm_flash))
    g.add_node("reflect",      partial(reflect,  llm=llm_pro))
    g.add_node("prepare_retry", prepare_retry)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", _route_after_classify, {
        "retrieve": "retrieve",
        "planner":  "planner",
    })
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "refine")
    g.add_conditional_edges("refine", _should_continue, {"retrieve": "retrieve", "answer": "answer"})
    g.add_conditional_edges("answer", _make_should_reflect_or_end(use_reflect), {"end": END, "reflect": "reflect"})
    g.add_conditional_edges("reflect", _should_reflect, {"end": END, "retrieve": "prepare_retry"})
    g.add_edge("prepare_retry", "retrieve")
    return g.compile()