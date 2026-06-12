from functools import partial

from langgraph.graph import END, START, StateGraph

from config import LLM_FLASH, LLM_PRO
from llm import get_llm
from nodes import answer, planner, retrieve
from state import AgentState


def _should_continue(state: AgentState) -> str:
    return "retrieve" if state["sub_questions"] else "answer"


def get_graph(tier: str = "flash"):
    """编译并返回图。

    tier: "flash"（默认）或 "pro"
    M1 模式：直接传入已填好 top_contexts 的 state（跳过 planner）。
    M2 模式：从 question + raw_contexts 出发，走完整 planner → retrieve 循环。
    """
    llm_flash = get_llm(model=LLM_FLASH)
    llm_pro   = get_llm(model=LLM_PRO if tier == "pro" else LLM_FLASH)

    g = StateGraph(AgentState)
    g.add_node("planner",  partial(planner, llm=llm_pro))
    g.add_node("retrieve", retrieve)
    g.add_node("answer",   partial(answer,  llm=llm_flash))

    g.add_edge(START, "planner")
    g.add_edge("planner", "retrieve")
    g.add_conditional_edges("retrieve", _should_continue, {"retrieve": "retrieve", "answer": "answer"})
    g.add_edge("answer", END)
    return g.compile()
