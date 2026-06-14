# P4 设计：+Reflection（反思回路）

## 目标

给医疗助手加 Agent 的 **Reflection 支柱**：answer 后让 LLM 自判「当前证据能否支撑一个**安全、
完整**的医疗回答」，不足则用具体的「缺失查询」补检索、再作答，带修订次数硬上限防死循环。

解决的问题：一次检索可能没覆盖问题的某个方面（如问「二甲双胍的副作用和禁忌」却只召回副作用），
答案不完整。反思回路让 Agent 自己发现缺口并补齐。

## 图结构（在 answer 后加回路）

```
... fuse → answer → reflect
                     ├─ 充分 / 达上限 → extract_memory → END
                     └─ 不足 → augment_retrieve → fuse → answer → reflect（回路）
```

- `reflect`：判断 evidence+answer 是否充分，输出 `missing_info`（不足时的具体补充查询）、`reflection`
  （理由），`revisions += 1`。
- `augment_retrieve`：用 `missing_info` 补检索内部库（若 `use_external` 也补外部），结果**追加去重**
  到 `internal_evidence`/`external_evidence`，再走 fuse→answer→reflect。
- `route_after_reflect`：`missing_info` 为空（充分）或 `revisions >= MAX_REVISIONS` → extract_memory；
  否则 → augment_retrieve。

## 关键取舍

| 取舍 | 决定 | 理由 |
|------|------|------|
| 证据累积 | augment **追加去重**而非覆盖，fuse 每轮对全部证据重排 | 覆盖会丢掉首轮证据；累积才能补全 |
| 防死循环 | `MAX_REVISIONS=2`（最多补检索 1 次）+ reflect prompt 克制 | 医疗不必无限补；prompt 强调「只在缺关键信息时判不足，不因可更详尽就重试」 |
| 补检索查询 | 用 reflect 给的具体 `missing_info`（含药名/方面） | 复述原问题会捞回同样证据，等于白补 |
| 对照开关 | `get_med_graph(use_reflect=False)` 退回无反思 | 便于对比反思的增量/开销 |
| 与记忆/模型兼容 | reflect/augment 在 answer 与 extract_memory 之间 | 不影响短期/长期记忆；模型仍按 config.model_tier 选 |

## 实现要点

- `med_state.py`：加 `revisions` / `reflection` / `missing_info`（无 reducer，节点 return 覆盖）。
- `med_nodes.py`：`reflect`（Pydantic `ReflectResult{sufficient, reasoning, missing}`，失败 try/except
  视为充分）、`augment_retrieve`（复用 `kb_search.search` + `external.web_search`，按 text 去重追加）、
  `route_after_reflect`。
- `med_graph.py`：`get_med_graph(use_reflect=True)` 插入两节点 + 条件边回路。
- 展示：`server` 的 `ChatResponse` 加 `revisions`；前端 `revisions>1` 显示「🔁 经过 N 轮证据反思核验」；
  CLI 打印反思轮数与理由。

## 验证结论（已端到端通过）

问「二甲双胍的副作用、禁忌症和漏服了怎么处理」（flash）：
- `revisions=2`（触发一次补检索）。
- `reflection`：「资料详细描述了副作用和禁忌症，但未涉及漏服处理方法，这是问题明确要求的关键方面」。
- `missing_info`：「二甲双胍漏服的处理方法」。
- 内部证据由首轮增至 10 条（补检索追加），最终答案副作用/禁忌/漏服均覆盖。
- 未死循环（达上限即停）。

## 待办 / 局限

- reflect/augment 每轮多次 LLM + 检索调用，增加延迟与成本（这是「质量换效率」的固有权衡）。
- 当前医疗轨道无 EM/F1 客观评测，反思增量靠定性 + recall@k / LLM-as-judge（见 `P1_DESIGN.md`）。
- 补检索仅依据单条 `missing_info`；多缺口可扩展为多查询（路线图）。
