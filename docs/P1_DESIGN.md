# P1 设计：Agent 接入 Qdrant 医疗知识库（医疗 RAG 基线）

## 目标与定位

P1 = **医疗轨道的基线**，地位等同 HotpotQA 轨道的 M1：

> 用户提问 → 从 **Qdrant 医疗库混合检索** → DeepSeek 综合证据作答 → **带来源引用**。

这是医疗 Agent 的"单跳基线"，后续逐层加 Planning / 外部 API 融合 / Reflection / Memory。

## 为什么独立一条轨道（不改 HotpotQA 的图）

现有 `graph.py / nodes.py / state.py` 是为 HotpotQA 设计的：

- 输入是**每题临时的 `raw_contexts`（10 段）**，retrieve 在这 10 段里重排。
- 有 `classify`（bridge/comparison 分流）——这是 HotpotQA 题型，医疗问诊没有。

医疗轨道的输入是**持久 Qdrant 库**（不是临时段落），题型也不同。所以**新开一组文件**，
HotpotQA 轨道保持不动（已 tag `v1.0-hotpotqa`，撑客观 EM/F1 指标），两轨并行。

## P1 图结构

```
question → retrieve(KB 混合检索) → answer(带引用) → END
```

最简单的单跳。对照后续：

```
P1  : question → retrieve → answer                                  ← 现在做这个
P2  : question → plan → retrieve → (refine) → answer                +Planning
P3  : question → plan → route → [KB / 外部API] → fuse → answer       +Tool Use/知识融合
P4  : ... → answer → reflect →（不足则换源补检索）→ END               +Reflection
P5  : +短期 checkpointer（多轮）+ 长期 Store（用户/经验）             +Memory
```

## 新增文件

| 文件 | 作用 |
|------|------|
| `med_state.py` | 医疗轨道图状态 |
| `med_nodes.py` | retrieve（查 KB）、answer（带引用） |
| `med_graph.py` | 组装 `question → retrieve → answer` |
| `med_rag.py`（可选） | CLI 入口：`python med_rag.py "高血压怎么治"` |

复用：`kb_search.py`（混合检索）、`llm.py`（DeepSeek）、`vectordb.py`、`embeddings.py`。

## 状态设计（med_state.py）

```python
class MedState(TypedDict):
    question: str
    evidence: list[dict]   # [{text, source_id, chunk_id, score}, ...]
    answer: str
    citations: list[str]   # 答案引用到的来源标识
```

## 节点设计（med_nodes.py）

### retrieve —— 查 KB（混合检索）

复用 `kb_search` 的混合召回 + 重排，但要**带回元数据**（source_id 等）供引用。
需把 `kb_search.hybrid_recall` 改成返回 payload（不仅是 text）：

```python
def retrieve(state: MedState) -> dict:
    hits = kb_search.hybrid_search_with_meta(state["question"])  # 返回带 source_id 的证据
    return {"evidence": hits}   # [{text, source_id, chunk_id, score}]
```

> 改动点：`kb_search.py` 加一个返回 payload 的版本（当前 `search()` 只回 {score, text}）。

### answer —— 带引用作答

把证据编号 `[1][2][3]` 喂给 DeepSeek，要求**引用编号**，再把编号映射回 source：

```python
_ANSWER_PROMPT = """你是严谨的医疗知识助手。只根据以下资料回答，不编造。
在关键结论后用 [编号] 标注依据。若资料不足以回答，明说"资料不足"。

资料：
{context}

问题：{question}
答案："""

def answer(state, *, llm):
    ctx = "\n\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(state["evidence"]))
    msg = llm.invoke([("system", _SYSTEM), ("human", _ANSWER_PROMPT.format(context=ctx, question=state["question"]))])
    return {"answer": msg.content.strip(), "citations": [...]}  # 解析出引用编号→source
```

> 医疗高风险，prompt 要求"资料不足明说"，且最终对外加免责声明"仅供学习演示，非医疗建议"。

## 图组装（med_graph.py）

```python
g = StateGraph(MedState)
g.add_node("retrieve", retrieve)
g.add_node("answer", partial(answer, llm=get_llm()))
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "answer")
g.add_edge("answer", END)
```

## 评测（P1 阶段先定性，指标后补）

- 评测集：`medical_data/finetune/test_zh_0.json`（instruction=患者问题，output=医生答复）。
- **医疗答复是开放长文本，EM 基本失效**（已在 M3 学到教训）。P1 先**定性**跑几条看效果；
  量化指标留给后续：**检索 recall@k**（金标 chunk 是否被召回）+ **LLM-as-judge** 评答案质量。
- 全量灌库完成后才好评测（库要全）。

## 改动清单（实现 P1 时）

1. `kb_search.py`：加 `hybrid_search_with_meta()`，召回带 source_id/chunk_id 的 payload。
2. 新建 `med_state.py` / `med_nodes.py` / `med_graph.py`。
3. （可选）`med_rag.py` CLI 入口，跑通 `问题→检索→带引用作答`。
4. 跑几条真实问诊问题定性验证。

## P1 完成标志

输入一个医疗问题（如"高血压的一线治疗药物有哪些"），agent 能：
检索到相关教材段 → 综合作答 → 标注 [来源] → 附免责声明。即医疗 RAG 端到端跑通。
