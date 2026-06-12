"""M0 冒烟测试：在写任何 agent 逻辑之前，先证明四条管道通了。

前置：
  - .env 里已填 DEEPSEEK_API_KEY
  - SSH 隧道已起（本机 9997 -> AutoDL Xinference），BGE 向量与重排模型已启动

运行：python smoke_test.py
期望：五行 [LLM]/[EMBED]/[RERANK]/[DATA]/[GRAPH] 都打印，且没有 [FAIL]。
"""
from typing import TypedDict

from data import load_hotpotqa
from embeddings import get_embeddings, rerank
from llm import get_llm


def check_llm():
    llm = get_llm()
    msg = llm.invoke(
        [
            ("system", "你是一个简洁、严谨的助手。"),
            ("human", "用一句话说明什么是向量检索。"),
        ]
    )
    print("[LLM] DeepSeek ->", msg.content[:160])


def check_embeddings():
    emb = get_embeddings()
    vec = emb.embed_query("测试一下中文向量化")
    print(f"[EMBED] BGE 维度={len(vec)}  前三维={[round(x, 4) for x in vec[:3]]}")


def check_rerank():
    query = "什么是过拟合？"
    docs = [
        "过拟合指模型在训练集上表现很好但泛化能力差。",
        "今天天气不错，适合出去散步。",
        "学习率是梯度下降的一个超参数。",
    ]
    res = rerank(query, docs, top_n=3)
    top = res[0]
    print(f"[RERANK] 最相关 -> {docs[top['index']]}  分数={top['relevance_score']:.4f}")


def check_dataset():
    ex = load_hotpotqa(n=1)[0]
    print(f"[DATA] HotpotQA 已加载  Q: {ex['question']}")
    print(
        f"        A: {ex['answer']}  "
        f"context 段数: {len(ex['context']['title'])}  "
        f"金标条数: {len(ex['supporting_facts']['title'])}"
    )


def check_langgraph():
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        x: int

    g = StateGraph(S)
    g.add_node("inc", lambda s: {"x": s["x"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    app = g.compile()
    print("[GRAPH] LangGraph OK，1+1 ->", app.invoke({"x": 1})["x"])


if __name__ == "__main__":
    for fn in (check_llm, check_embeddings, check_rerank, check_dataset, check_langgraph):
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
