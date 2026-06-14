# P5 设计：+Memory（短期多轮 + 长期用户记忆）

## 目标

给医疗助手加 Agent 的 **Memory 支柱**：
- **短期记忆（会话内多轮）**：记住本次对话上下文，支持指代消解（「那它的禁忌呢」→「二甲双胍的禁忌」）。
- **长期用户记忆（跨会话）**：记住用户健康背景（过敏史/慢病/在用药），作答时召回并做个性化 + 安全提示。

## 为什么是「用户记忆」而非「事实缓存」

CLAUDE.md 原列了「事实缓存 + 经验记忆」，但医疗场景下选了**用户记忆**优先：
- 医疗最有价值的记忆是过敏史/慢病/在用药——直接影响用药建议**安全性**。
- 事实缓存（缓存问答省检索）在医疗有坑：知识有时效、同病不同人答案不同、缓存旧答案有风险；
  其卖点「检索次数下降」是工程优化，排在正确性/安全性之后。故列入路线图，本期不做。

## 图结构

```
question
  → contextualize   [短期] history 非空时用最近 N 轮把问题改写成自包含 standalone_question；首轮跳过
  → recall_memory   [长期·读] 按 user_id 从 Store 语义召回相关健康事实
  → plan → retrieve_internal → retrieve_external → fuse
  → answer          注入 user_memory + 证据作答；把本轮 (问,答) 追加进 history
  → extract_memory  [长期·写] LLM 从本轮用户输入抽取健康事实，写入 Store
  → END
```

- `recall_memory` 在作答前（要用召回结果）；`extract_memory` 放最后（不影响作答内容；略增延迟，
  可优化为后台异步，本期求正确先放图里）。
- `contextualize` 改写后的 `standalone_question` 贯穿下游 plan/retrieve/fuse/answer，
  下游逻辑几乎不动。

## 技术实现（LangGraph）

| 能力 | 实现 | 关键点 |
|------|------|--------|
| 短期 | `InMemorySaver` checkpointer | 按 `thread_id` 持久化整个 state；`history` 字段用 `Annotated[list, operator.add]` reducer 跨轮累积 |
| 长期 | `InMemoryStore(index={embed, dims, fields})` | `embed` 直接传现成 `get_embeddings()`（BGE），`dims=EMBED_DIM=1024`；按 `("memories", user_id)` namespace 隔离 |
| 读 | `store.search(ns, query=..., limit=K)` | BGE 语义检索召回相关健康事实 |
| 写 | `store.put(ns, uuid, {"text": fact})` | `extract_memory` 用结构化输出（Pydantic `HealthFacts`）抽取 |
| 注入 | 节点签名 `store: BaseStore` + `config: RunnableConfig` | LangGraph 运行时自动注入；`user_id`/`thread_id` 走 `config.configurable` 不进 state |

挂载：`med_graph.py` 的 `compile(checkpointer=..., store=...)`；
invoke：`config={"configurable": {"thread_id": session_id, "user_id": user_id}}`。

## 防误抽 / 防幻觉

- `extract_memory` prompt 严格约束：只记用户**明确陈述**的过敏/慢病/长期用药，没有就返回空，
  不臆测、不把一次性症状当长期事实。抽取失败 try/except 跳过，不影响作答。
- `answer` 系统提示：若用药建议与用户过敏史/禁忌冲突，必须主动提示并给替代方向。

## 前端 / 接口

- `server.py`：`/chat` 收 `session_id`（→ thread_id）+ `user_id`（→ Store namespace），响应加 `standalone_question`。
- `static/index.html`：聊天式气泡累积；`localStorage` 存 `user_id`（长期不变）+ `session_id`（「新会话」按钮重置）。
  新会话换 session_id 清空气泡，但 user_id 不变 → 长期记忆仍在。
- `med_rag.py --chat`：CLI 多轮交互（固定 user_id + thread_id）。

## 验证结论（已端到端通过）

1. **指代消解**：「二甲双胍的副作用」→「那它的禁忌呢」→ standalone 改写为「二甲双胍的禁忌」。✓
2. **自动抽取**：「我有二型糖尿病，对青霉素过敏」→ 写入 `['对青霉素过敏','患有二型糖尿病']`。✓
3. **跨会话召回 + 安全提示**：新 session 同 user 问「嗓子发炎吃什么消炎药」→ 召回过敏史，
   答案主动提示「有青霉素过敏史，若处方抗生素须避开青霉素类」。✓
4. 容器版（`/chat` 带 session_id）多轮链路同样通过。

## 取舍与待办

- **内存版**：`InMemorySaver` + `InMemoryStore`，进程内有效、跨会话（同进程不同 thread）有效、重启清空。
  演示足够；生产持久化 → `SqliteSaver` + Qdrant-backed `BaseStore`（路线图）。
- `extract_memory` 同步在图里，略增每轮延迟 → 可优化为返回后后台执行。
- 长期记忆去重/冲突处理（同一事实多次陈述会重复写）→ 可加去重或 upsert（路线图）。
