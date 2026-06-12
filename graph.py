from functools import partial

from langgraph.graph import END, START, StateGraph

from config import LLM_FLASH, LLM_PRO
from llm import get_llm
from nodes import answer, retrieve
from state import AgentState


def get_graph(tier: str = "flash"):
    """编译并返回 M1 基线图。

    tier: "flash"（默认，快速便宜）或 "pro"（更强推理）
    后续 UI 按钮切换时传这个参数即可，其余代码不用动。
    """
    model = LLM_PRO if tier == "pro" else LLM_FLASH
    llm = get_llm(model=model)

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", partial(answer, llm=llm))
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()