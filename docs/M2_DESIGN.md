# M2 设计决策

## 目标

在 M1 单跳 RAG 基线上加入 Planning 层，让 Agent 能把多跳问题拆解成子问题分步检索，重点提升 HotpotQA bridge 子集的 EM/F1。

## 图结构选择

### 方案 A：LangGraph 条件边回路（采用）

```
planner → retrieve → refine → should_continue → retrieve（循环）
                                             └→ answer
```

planner 一次生成所有子问题列表，retrieve 每次弹出队列头部执行检索，refine 提取中间答案并精化下一个子问题，should_continue 判断队列是否为空决定继续循环还是进入 answer。

**选择理由：**
- Planning 在图拓扑上可见，符合 Agent 范式，LangSmith 中每步均可追踪
- 条件边回路天然支持后续插入 Reflection 节点（M3），无需重构
- 简历叙事上更有说服力

### 方案 B：retrieve 内部 for 循环（未采用）

retrieve 节点内部遍历所有子问题，图结构与 M1 相同。实现简单但 Planning 对外不可见，且 M3 加 Reflection 时需要重新设计图。

## 静态子问题 vs 链式精化

### 静态子问题（M2-static，已验证）

planner 一次性生成全部子问题，子问题之间没有依赖：

```
SubQ1: "谁在 Kiss and Tell 里演了 Corliss Archer？"
SubQ2: "那位演员担任过什么政府职位？"   ← 仍是泛指
```

retrieve 用 SubQ2 的原始文本检索，语义与原问题几乎相同，证据池没有实质变化。

**结果（n=200）：**
- total  EM=0.5250  F1=0.6656
- bridge EM=0.5241（与 M1 持平）
- comparison EM=0.5294（较 M1 下跌 0.18）

comparison 下跌原因：M1 一次喂入所有上下文更利于跨实体对比；拆成子问题后证据被切割，反而丢失连贯性。

### 链式精化（M2-refine，当前版本）

每次 retrieve 后，refine 节点从最新证据中提取中间答案，代入下一个子问题：

```
检索 SubQ1 → 证据中提取中间答案 "Shirley Temple"
SubQ2 精化：
  精化前："那位演员担任过什么政府职位？"
  精化后："Shirley Temple 担任过什么政府职位？"   ← 具体名字
检索精化后的 SubQ2 → 命中更精准的证据
```

| | 静态 | 链式精化 |
|---|---|---|
| 子问题何时确定 | 一次全部生成 | 边检索边精化 |
| 子问题间依赖 | 无 | 上一跳答案填入下一跳 |
| 对 bridge 题 | 与 M1 持平（EM=0.5241） | **+0.036 vs M1（EM=0.5602）** |
| 对 comparison 题 | 下跌（EM=0.5294） | 仍低于 M1（EM=0.5882 vs 0.7059） |

## 意图分类实验（已回退）

尝试让 planner 判断 comparison/bridge 类型，comparison 题保持单子问题路径。

实验结果（n=200）：

| 版本 | total EM | bridge EM | comparison EM |
|---|---|---|---|
| M1 基线 | 0.5550 | 0.5241 | 0.7059 |
| M2-refine（采用） | **0.5650** | 0.5602 | 0.5882 |
| +意图分类 | 0.5700 | **0.5904** | 0.4706 |
| +意图分类+格式修复 | 0.5400 | 0.5361 | 0.5588 |

**结论**：comparison 题的根本问题是 M2 的 planner 架构引入了额外噪声，无论如何修复都无法还原 M1 的 0.7059。已回退至 M2-refine 版本，comparison 的损失作为已知局限记录，留待 M3 统一处理。

## planner 结构化输出

DeepSeek v4-flash 的 thinking 模式不支持 tool_choice，因此 `with_structured_output` 必须使用 `method="json_mode"` 而非默认的函数调用模式：

```python
llm.with_structured_output(SubQuestions, method="json_mode")
```

## refine 节点设计

单次 LLM 调用同时完成两件事（减少 API 调用次数）：
1. 从最新证据中提取上一跳的简短答案
2. 将答案代入下一个子问题，输出精化后的查询

输出结构：`{"intermediate_answer": "...", "refined_question": "..."}`

## top-k

沿用 M1 的 top-3，每个子问题各取 top-3 段落追加到 evidence。