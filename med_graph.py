"""医疗 RAG 图（P3：知识融合）。

question → plan → retrieve_internal(Qdrant 教材)
               → retrieve_external(Tavily Web)
               → fuse(合并去重重排) → answer(带来源引用) → END

内外部异构源融合，正是 JD 的"对接企业知识库/外部 API + 知识融合"。
复用 kb_search / external / llm。后续加 Reflection / Memory。
"""
from functools import partial

from langgraph.graph import END, START, StateGraph

from config import LLM_FLASH
from llm import get_llm
from med_nodes import answer, fuse, plan, retrieve_external, retrieve_internal
from med_state import MedState


def get_med_graph(tier: str = "flash"):
    llm = get_llm(model=LLM_FLASH)
    g = StateGraph(MedState)
    g.add_node("plan", partial(plan, llm=llm))
    g.add_node("retrieve_internal", retrieve_internal)
    g.add_node("retrieve_external", retrieve_external)
    g.add_node("fuse", fuse)
    g.add_node("answer", partial(answer, llm=llm))
    g.add_edge(START, "plan")
    g.add_edge("plan", "retrieve_internal")
    g.add_edge("retrieve_internal", "retrieve_external")
    g.add_edge("retrieve_external", "fuse")
    g.add_edge("fuse", "answer")
    g.add_edge("answer", END)
    return g.compile()
