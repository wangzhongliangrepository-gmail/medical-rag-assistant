"""医疗 RAG 图（P3 知识融合 + P5 记忆）。

contextualize → recall_memory → plan → retrieve_internal → retrieve_external
             → fuse → answer → extract_memory → END

短期记忆：InMemorySaver（checkpointer），按 thread_id 持久化会话内 state（含 history）。
长期记忆：InMemoryStore（按 user_id namespace），用 BGE 做语义检索，存用户健康事实。
两者进程级单例，随编译图存活；重启清空（生产可换 SqliteSaver + Qdrant-backed Store）。
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from config import EMBED_DIM
from embeddings import get_embeddings
from med_nodes import (
    answer,
    augment_retrieve,
    contextualize,
    extract_memory,
    fuse,
    plan,
    recall_memory,
    reflect,
    retrieve_external,
    retrieve_internal,
    route_after_reflect,
)
from med_state import MedState


def get_med_graph(use_reflect: bool = True):
    """use_reflect 是反思回路的开关：默认 True 走完整反思；传 False 退回「答完直接结束」
    use_reflect=False：退回无反思路径（answer → extract_memory 直连），作对照开关。
    模型按请求动态选（节点内从 config 的 model_tier 取 flash/pro）；checkpointer/store 单例共享。
    """
    checkpointer = InMemorySaver()  # 短期：会话内多轮
    store = InMemoryStore(index={   # 长期：跨会话用户记忆（BGE 语义检索）
        "embed": get_embeddings(),  # 用 BGE 把记忆向量化
        "dims": EMBED_DIM,          # 向量维度 1024
        "fields": ["text"],         # 对哪个字段建语义索引
    })

    g = StateGraph(MedState)                                # 建一张以 MedState 为状态的空图
    g.add_node("contextualize", contextualize)
    g.add_node("recall_memory", recall_memory)
    g.add_node("plan", plan)
    g.add_node("retrieve_internal", retrieve_internal)
    g.add_node("retrieve_external", retrieve_external)
    g.add_node("fuse", fuse)
    g.add_node("answer", answer)
    g.add_node("extract_memory", extract_memory)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "recall_memory")
    g.add_edge("recall_memory", "plan")
    g.add_edge("plan", "retrieve_internal")
    g.add_edge("retrieve_internal", "retrieve_external")
    g.add_edge("retrieve_external", "fuse")
    g.add_edge("fuse", "answer")

    if use_reflect:
        # 反思回路：answer → reflect →（充分/达上限 → extract_memory / 不足 → augment_retrieve → fuse → answer）
        g.add_node("reflect", reflect)
        g.add_node("augment_retrieve", augment_retrieve)
        g.add_edge("answer", "reflect")
        g.add_conditional_edges("reflect", route_after_reflect, {
            "extract_memory": "extract_memory",
            "augment_retrieve": "augment_retrieve",
        })
        g.add_edge("augment_retrieve", "fuse")
    else:
        g.add_edge("answer", "extract_memory")

    g.add_edge("extract_memory", END)

    return g.compile(checkpointer=checkpointer, store=store)
