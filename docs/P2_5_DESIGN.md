# P2.5 设计：多跳（桥接型）查询 —— 链式精化

## 问题：桥接型 ≠ 平行型

P2 的 Planning 是**平行拆解**——把问题拆成互相独立的方面，同时检索。但有一类问题是
**桥接型（多跳）**，后一跳依赖前一跳的答案：

```
平行型（P2 已处理）："二甲双胍的适应症、副作用、禁忌"
  → 三方面独立，可同时检索

桥接型（本设计）："治疗高血压的一线药的副作用是什么"
  → Hop1 先查"一线药是谁" → Hop2 才能查"它的副作用"
```

桥接型若平行拆，第二个子问题"副作用是什么"是**悬空的**（副作用 of 谁？），检索全是噪声。

## 核心机制：链式精化（chain refinement）

```
Hop1: "高血压的一线治疗药物是什么"
   → 检索 → 从证据提取中间答案："钙通道阻滞药、ACEI、利尿药"
Hop2: "[上一步的药]的副作用"
   → refine 把中间答案代入 → "钙通道阻滞药、ACEI、利尿药的副作用"
   → 用改写后的查询再检索 → 拿到副作用证据
   → 综合作答
```

关键是 **`refine` 节点**：从刚检索到的证据**抽取中间答案**，再**代入下一个子问题**锚定它。

## 图结构（带回路，与 HotpotQA M2-refine 同构）

```
question → plan → retrieve → refine ──┐ 还有子问题
                    ▲                 │
                    └─────────────────┘
                                      │ 子问题用完
                                      ▼
                                    answer → END
```

与 P2 的区别：**P2 一次性检索所有子问题；多跳是顺序循环**——检索队首 → 精化下一个
→ 再检索，逐跳推进。

## 节点设计

| 节点 | 改动 |
|------|------|
| `plan` | 输出**有序**子问题，后面的可引用"上一步结果" |
| `retrieve` | 只处理**队首**子问题（消费队列），累积证据 |
| **`refine`**（新增） | 从当前证据抽中间答案 → 用它改写下一个子问题（代入锚定） |
| 条件边 | 还有子问题 → 回 `retrieve`；否则 → `answer` |

## 状态变更（med_state.py）

```python
class MedState(TypedDict):
    question: str
    sub_questions: list[str]        # 当队列消费（处理一个弹一个）
    last_sub_question: str          # retrieve 刚处理的子问题（供 refine）
    intermediate_answers: list[str] # 每跳提取的中间答案
    evidence: list[dict]            # 累积去重
    answer: str
```

## refine 节点逻辑（med_nodes.py）

```python
class RefineResult(BaseModel):
    intermediate_answer: str   # 上一跳证据里的关键答案（药名/实体）
    refined_question: str      # 把它代入后改写的下一子问题

def refine(state, *, llm):
    if not state["sub_questions"]:      # 没有下一跳，跳过
        return {}
    # 从最近证据 + last_sub_question 提取中间答案，改写 sub_questions[0]
    ...
    return {
        "intermediate_answers": [...] + [result.intermediate_answer],
        "sub_questions": [result.refined_question] + state["sub_questions"][1:],
    }
```

## 区分 bridge / parallel：让 refine 自适应（方案 A）

不单独做 classify，让 refine **永远尝试**"用已知中间答案改写下一子问题"：

- **桥接型**：下一子问题含"该药/上述/它的"等指代 → 成功代入锚定。
- **平行型**：下一子问题已自包含（"二甲双胍的副作用"）→ 改写基本是 no-op，无害。

> 备选方案 B：先 classify 判 bridge/parallel/single 再路由（更精确、更复杂）。先用 A。

## 必须记住的教训（来自 HotpotQA 轨道）

- 链式精化对 bridge **+0.036 EM**（有效）。
- 但**绝不能强加给非桥接问题**：HotpotQA 强拆 comparison 掉了 0.07 EM。
- 故 refine 必须自适应 no-op（方案 A），或靠 classify 兜住（方案 B），不能盲目链式。

## 改动清单

1. `med_state.py`：加 `last_sub_question`、`intermediate_answers`。
2. `med_nodes.py`：加 `refine`；`retrieve` 改为消费队首子问题、记录 `last_sub_question`。
3. `med_graph.py`：retrieve → refine → 条件边（回 retrieve / 去 answer）。
4. 验证："治疗高血压的一线药的副作用" → Hop1 查药 → refine 代入 → Hop2 查副作用。

## 完成标志

桥接型问题能：第一跳检索出中间实体 → refine 正确代入改写 → 第二跳锚定检索 →
综合作答。且平行型问题（P2 那类）不被破坏。
