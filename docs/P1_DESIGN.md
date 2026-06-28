# P1 设计：医疗 RAG 基线（带引用作答）

## 目标

在 [P0](P0_DESIGN.md) 检索底座上，搭出最小可用的医疗问答闭环：
**问诊 → 混合检索 → 由 LLM 结合证据作答**，且作答**只依据资料、强制 `[编号]` 引用、防幻觉**。

## 为什么把「可溯源」当硬约束

医疗是高风险场景，答案必须**能查到出处**、**不能编**。所以基线就定下三条铁律，贯穿后续所有阶段：

1. **只用检索到的资料**作答，资料里没有的不说。
2. **关键结论后标 `[编号]`**，编号对应证据列表，可点开溯源。
3. **资料不足如实说明**，宁可说"无法回答"也不臆测；对外附免责声明「仅供学习演示，非医疗建议」。

## 实现（`med_state.py` / `med_nodes.py` / `med_graph.py`）

用 **LangGraph `StateGraph`** 串起节点，状态是 `MedState`（TypedDict）。基线主干：

```
question → retrieve_internal → fuse → answer → END
```

| 节点 | 职责 |
|------|------|
| `retrieve_internal` | 调 `kb_search.search()` 混合检索，证据标来源 `内部·教材#{source_id}` |
| `fuse` | 基线期只对内部证据按问题统一重排取 top-k（为 [P3](P3_DESIGN.md) 融合外部留好接口） |
| `answer` | 把证据拼成 `[i]（来源）正文`，喂给 DeepSeek，带 `[编号]` 引用作答 |

**作答约束落在 prompt 上**（`_SYSTEM` + `_ANSWER_PROMPT`）：系统提示「只依据资料、绝不编造、
不足如实说明」；用户提示要求「`[编号]` 标注依据、冲突指出分歧、资料不足明说」。

**防幻觉兜底**：`answer` 节点发现 `evidence` 为空时，直接返回「根据现有资料无法回答」，
不让 LLM 自由发挥。

## 关键设计取舍

- **LLM 只做「读资料 + 组织语言 + 标引用」**，不靠它的参数化知识答医疗问题——把幻觉面收到最小。
- **DeepSeek `.content` 干净**：思考内容在独立的 `reasoning_content` 字段，不需要剥 `<think>`。
- **稳定系统提示放前部**：命中上下文缓存降本（DeepSeek 上下文缓存默认开）。
- **状态字段一次设计到位**：`MedState` 里预留了 `sub_questions`/`external_evidence`/`evidence` 等，
  后续 P2/P3/P4 加节点几乎不动已有节点。

## 验证

医疗答复是开放长文本，EM/F1 失效 → 用 **LLM-as-judge** 评「是否忠于证据、引用是否对应、
有无幻觉」。检索侧的 recall@k 在 [P0](P0_DESIGN.md)。

## 待办 → 后续阶段

- 复合问诊（一问多面）只检索一次会漏 → [P2 Planning](P2_DESIGN.md)。
- 教材是稳定知识、缺时效 → [P3 知识融合](P3_DESIGN.md) 接外部 Web。
- 证据不足时不会自救 → [P4 Reflection](P4_DESIGN.md)。

