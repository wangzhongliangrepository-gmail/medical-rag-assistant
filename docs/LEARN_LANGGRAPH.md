# 学习串讲：LangGraph 基础 + 和 LangChain 的区别

> 简历技能点。结合本项目代码讲：什么是 LangGraph、和 LangChain 区别、怎样算"熟练"。

## 一、一句话区别

> **LangChain 是「积木」（组件 + 线性链），LangGraph 是「把积木按带循环的流程图组装起来的引擎」。**

- **LangChain**：提供 LLM 封装、检索器、解析器、prompt 模板等**组件**，以及 LCEL
  （`prompt | llm | parser` 链式串联）。擅长**线性 / DAG** 流程。
- **LangGraph**：把应用建模成**状态图（StateGraph）**，专门处理**有状态、多步、会循环、要
  分支**的 Agent。解决 LangChain 链表达不了的：**循环、条件跳转、持久化状态、人机交互**。

| | LangChain（LCEL 链） | LangGraph（状态图） |
|--|--|--|
| 控制流 | 线性 / DAG，单向 | **支持循环、条件分支** |
| 状态 | 链里数据流过就完 | **显式 State，跨节点共享、可持久化** |
| 循环（如反思重试） | 很难优雅表达 | **天生支持**（条件边回到前面节点） |
| 记忆 | 要自己接 | **内置 checkpointer（短期）+ Store（长期）** |
| 适合 | 固定流程 RAG、单次调用 | 真正的 Agent（多步、自我纠错、多轮） |

**关系**：同一团队、同一生态。LangGraph 负责**编排/控制流**，节点内部常用 LangChain 组件。
本项目就是这么配合：

```python
from langgraph.graph import StateGraph       # LangGraph：编排
from langchain_deepseek import ChatDeepSeek  # LangChain：LLM 组件
llm.with_structured_output(...)              # LangChain：结构化输出
```

## 二、为什么本项目必须用 LangGraph（关键面试答案）

有三样东西纯 LCEL 链做不优雅：

1. **反思回路是「循环」**：`answer → reflect →（不够）→ augment_retrieve → fuse → answer`，
   带终止条件的环。LCEL 单向，表达不了"回去重来 N 次"。
2. **条件分支**：`reflect` 后「充分→收尾 / 不足→补检索」靠**条件边**。
3. **持久化状态/记忆**：多轮靠 `checkpointer` 按 `thread_id` 存取 state，长期记忆靠 `Store`。

> 一句话：**因为我的 Agent 需要循环（反思重试）、条件分支、跨轮记忆，这些正是 LangGraph
> 相对 LangChain 链的核心价值。**

## 三、核心知识点（对照本项目代码）

| 概念 | 是什么 | 本项目 |
|------|--------|--------|
| **StateGraph** | 状态图本体 | `g = StateGraph(MedState)` |
| **State** | 节点间共享的数据结构 | `MedState`（TypedDict）|
| **Reducer** | 字段如何合并（覆盖 or 累加） | `history: Annotated[list, operator.add]`（累加）|
| **Node** | 处理函数 `(state)→部分state` | `plan` / `retrieve` / `answer` 等 |
| **Edge** | 固定连接 A→B | `g.add_edge("plan", "retrieve_internal")` |
| **Conditional Edge** | 按 state 决定走向 | `add_conditional_edges("reflect", route_after_reflect, {...})` |
| **compile** | 编译成可执行图 | `g.compile(checkpointer=..., store=...)` |
| **checkpointer / Store** | 短期 / 长期记忆 | `InMemorySaver` / `InMemoryStore` |

进阶（也用了，加分项）：
- **循环 + 终止条件**：反思回路 + `MAX_REVISIONS` 防死循环。
- **节点依赖注入**：节点签名写 `config: RunnableConfig`、`store: BaseStore`，运行时自动注入。
- **运行时配置**：`invoke(..., config={"configurable": {"thread_id":..., "user_id":..., "model_tier":...}})`。

## 四、"熟练"分级（对照本项目达到哪档）

- **入门**：会 StateGraph 串节点、add_edge、compile、invoke（线性图）。
- **中级**：条件边分支、循环（带终止）、State + reducer、结构化输出节点。
- **熟练（简历可写）**：
  - ✅ checkpointer 多轮/持久化（按 thread_id）
  - ✅ Store 长期记忆（语义检索）
  - ✅ 节点注入 config/store、运行时传配置
  - ✅ 循环 + 防死循环、条件路由
  - ✅ 错误处理/优雅降级（try/except）

**结论**：本项目已覆盖「熟练」档绝大部分（条件边、反思循环+终止、Annotated reducer、
checkpointer 多轮、Store 语义长期记忆、依赖注入、动态选模型、结构化输出、优雅降级）。
简历写"熟练使用 LangGraph"撑得住——前提是能对着项目讲出来。

## 五、还没碰、但该知道边界（防被问倒）

本项目未用，但被问到要能说"知道，没在这个项目用上"：
- **streaming**：`graph.stream()` 流式输出中间结果。
- **human-in-the-loop / interrupt**：图中途暂停等人工确认。
- **subgraph（子图）**：一组节点封装成可复用子图。
- **持久化后端**：`SqliteSaver` / `PostgresSaver` 替代内存版（路线图）。

## 六、面试问题清单

1. LangGraph 和 LangChain 区别？→ 组件/线性链 vs 状态图/循环+分支+持久化。
2. 为什么用 LangGraph 不用 LangChain 链？→ 需要反思循环、条件分支、跨轮记忆。
3. State 怎么传递？reducer 干嘛？→ 共享 TypedDict；reducer 决定字段合并方式（覆盖 vs 累加）。
4. 条件边怎么用？→ 函数读 state 返回下一节点名，如 `route_after_reflect`。
5. 怎么循环又不死循环？→ 条件边回前面节点 + 计数上限（`MAX_REVISIONS`）。
6. 多轮怎么记上下文？→ checkpointer + `thread_id`，state 自动存取。
7. 长期记忆怎么做？→ Store + 语义检索，按 user_id namespace。
8. 节点怎么拿 store/配置？→ 签名声明 `store`/`config`，运行时注入。

## 一句话总结

LangChain 给积木（LLM、检索器、解析器），LangGraph 给一个能**循环、分支、记状态**的引擎
来组装 Agent。本项目把条件边、反思循环、双层记忆、依赖注入都用上了，够"熟练"——关键是
能对着代码讲出来。
