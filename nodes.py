from config import RERANK_TOP_K
from embeddings import rerank
from state import AgentState

_SYSTEM = "你是一个严谨、简洁的问答助手。只根据所给资料回答，不要编造。"

_PROMPT = """\
根据以下资料，简洁地回答问题。只输出答案本身，不要解释。

{context}

问题：{question}
答案："""


def retrieve(state: AgentState) -> dict:
    docs = [
        c["title"] + " " + " ".join(c["sentences"])
        for c in state["raw_contexts"]
    ]
    results = rerank(state["question"], docs, top_n=RERANK_TOP_K)
    top = [
        f"[{i+1}] 标题：{state['raw_contexts'][r['index']]['title']}\n"
        f"内容：{' '.join(state['raw_contexts'][r['index']]['sentences'])}"
        for i, r in enumerate(results)
    ]
    return {"top_contexts": top}


def answer(state: AgentState, *, llm) -> dict:
    context = "\n\n".join(state["top_contexts"])
    prompt = _PROMPT.format(context=context, question=state["question"])
    msg = llm.invoke([("system", _SYSTEM), ("human", prompt)])
    return {"answer": msg.content.strip()}