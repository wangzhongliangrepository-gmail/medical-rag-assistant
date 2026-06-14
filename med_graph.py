"""医疗 RAG 图（P3 知识融合 + P5 记忆）。

contextualize → recall_memory → plan → retrieve_internal → retrieve_external
             → fuse → answer → extract_memory → END

短期记忆：InMemorySaver（checkpointer），按 thread_id 持久化会话内 state（含 history）。
长期记忆：InMemoryStore（按 user_id namespace），用 BGE 做语义检索，存用户健康事实。
两者进程级单例，随编译图存活；重启清空（生产可换 SqliteSaver + Qdrant-backed Store）。
"""
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from config import EMBED_DIM, LLM_FLASH
from embeddings import get_embeddings
from llm import get_llm
from med_nodes import (
    answer,
    contextualize,
    extract_memory,
    fuse,
    plan,
    recall_memory,
    retrieve_external,
    retrieve_internal,
)
from med_state import MedState


def get_med_graph(tier: str = "flash"):
    llm = get_llm(model=LLM_FLASH)

    checkpointer = InMemorySaver()  # 短期：会话内多轮
    store = InMemoryStore(index={   # 长期：跨会话用户记忆（BGE 语义检索）
        "embed": get_embeddings(),
        "dims": EMBED_DIM,
        "fields": ["text"],
    })

    g = StateGraph(MedState)
    g.add_node("contextualize", partial(contextualize, llm=llm))
    g.add_node("recall_memory", recall_memory)
    g.add_node("plan", partial(plan, llm=llm))
    g.add_node("retrieve_internal", retrieve_internal)
    g.add_node("retrieve_external", retrieve_external)
    g.add_node("fuse", fuse)
    g.add_node("answer", partial(answer, llm=llm))
    g.add_node("extract_memory", partial(extract_memory, llm=llm))

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "recall_memory")
    g.add_edge("recall_memory", "plan")
    g.add_edge("plan", "retrieve_internal")
    g.add_edge("retrieve_internal", "retrieve_external")
    g.add_edge("retrieve_external", "fuse")
    g.add_edge("fuse", "answer")
    g.add_edge("answer", "extract_memory")
    g.add_edge("extract_memory", END)

    return g.compile(checkpointer=checkpointer, store=store)
