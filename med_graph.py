"""医疗 RAG 图（P2：+Planning）。

question → plan(拆子问题) → retrieve(逐子问题查 KB 汇总) → answer(带引用) → END

独立于 HotpotQA 轨道（graph.py）。复用 kb_search / llm / vectordb。
后续逐层加 外部 API 融合 / Reflection / Memory。
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from config import LLM_FLASH
from llm import get_llm
from med_nodes import answer, plan, retrieve
from med_state import MedState


def get_med_graph(tier: str = "flash"):
    llm = get_llm(model=LLM_FLASH)
    g = StateGraph(MedState)
    g.add_node("plan", partial(plan, llm=llm))
    g.add_node("retrieve", retrieve)
    g.add_node("answer", partial(answer, llm=llm))
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)
    return g.compile()
