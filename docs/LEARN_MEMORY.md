# 学习串讲：Memory 三节点（短期读 / 长期读 / 长期写）

> 配合 `med_nodes.py`、`med_state.py`、`med_graph.py` 阅读。前置：thread_id vs user_id 隔离原理。

## 三节点定位

```
contextualize → recall_memory → plan → ... → answer → ... → extract_memory → END
  [短期·读历史]   [长期·读档案]              [写短期]        [长期·写档案]
```

| 节点 | 属于 | 干什么 |
|------|------|--------|
| `contextualize` | 短期记忆 | 读对话历史，把「它」改成具体实体（指代消解）|
| `recall_memory` | 长期记忆 | 按 user_id 调出用户健康档案 |
| `extract_memory` | 长期记忆 | 把本轮用户说的健康信息存进档案 |

短期记忆的「写」不在单独节点——`answer` 顺手把这轮问答追加进 history。

## ⭐ 题眼：短期和长期，代码写法完全不同

| | 短期记忆（history） | 长期记忆（健康档案） |
|--|--|--|
| 钥匙 | thread_id | user_id |
| 存储 | checkpointer | store |
| 代码怎么操作 | **隐式**——只管读写 `state["history"]`，LangGraph 按 thread_id **自动**存取整个 state | **显式**——自己调 `store.search()`/`store.put()`，自己用 user_id 做 namespace |
| 看得到钥匙吗 | 看不到 thread_id（背后自动） | 看得到 user_id（自己从 config 取）|

一句话：**短期记忆是 LangGraph 替你管的（你只动 state），长期记忆是你自己管的（自己 put/search）。**

## 节点 1：contextualize（短期·读历史做指代消解）

```python
def contextualize(state, config):
    history = state.get("history", [])              # ① 读历史（checkpointer 已自动恢复）
    if not history:                                  # ② 首轮没历史 → 原样返回
        return {"standalone_question": state["question"]}
    llm = _llm(config)
    recent = history[-2 * HISTORY_WINDOW:]          # ③ 只取最近几轮（控制上下文）
    history_text = "\n".join(f"{h['role']}：{h['content']}" for h in recent)
    msg = llm.invoke([("system", "你是指代消解器..."),
                      _CONTEXTUALIZE_PROMPT.format(history=history_text, question=state["question"])])
    return {"standalone_question": msg.content.strip() or state["question"]}  # ④ 输出改写后问题
```

- ① `state["history"]` 有上几轮内容，是因为 checkpointer 按 thread_id 自动存/恢复，代码里没碰 thread_id。
- ② 首轮 history 空 → 不调 LLM，原问题当 standalone_question。
- ③ `[-2*HISTORY_WINDOW:]` 只取最近 4 轮（一轮 user+assistant 两条，×2），防历史撑爆上下文。
- ④ 输出 `standalone_question`，下游 plan/检索全用它。

**history 怎么写进去**：`answer` 结尾 `return {"history": [{user...}, {assistant...}]}` 返回新增两条；
`med_state.py` 里 `history: Annotated[list, operator.add]` 用累加 reducer，LangGraph 自动追加（不覆盖），
checkpointer 持久化。这就是短期"隐式"的秘密：reducer 累加 + checkpointer 存，你只管 return 新增。

## 节点 2：recall_memory（长期·读档案）

```python
def recall_memory(state, config, *, store):          # ← store 自动注入
    q = state.get("standalone_question") or state["question"]
    items = store.search(_user_ns(config), query=q, limit=MEMORY_RECALL_K)  # 语义检索
    return {"user_memory": [it.value["text"] for it in items]}
```

- `*, store`：签名声明 store，LangGraph 运行时自动注入（config 同理）。
- `_user_ns(config)`：
  ```python
  def _user_ns(config):
      user_id = (config.get("configurable") or {}).get("user_id", "anonymous")
      return (MEMORY_NAMESPACE, user_id)   # 如 ("memories", "用户A")
  ```
  从 config 取 user_id 拼 namespace——长期记忆"显式"隔离，A 查不到 B。
- `store.search(ns, query=q, limit=3)`：在该用户 namespace 里用当前问题做 BGE 语义检索，召回 top-3。
  （语义能力来自 med_graph 里 store 配的 `index={"embed": BGE}`。）

## 节点 3：extract_memory（长期·写档案）

```python
def extract_memory(state, config, *, store):
    structured = _llm(config).with_structured_output(HealthFacts, method="json_mode")
    try:
        result = structured.invoke([("system", "你是健康信息抽取器..."),
                                    _EXTRACT_PROMPT.format(utterance=state["question"])])
    except Exception:
        return {}                                    # 抽取失败不影响作答
    ns = _user_ns(config)
    for fact in result.facts:
        if fact.strip():
            store.put(ns, str(uuid.uuid4()), {"text": fact})   # 逐条写入该用户档案
    return {}
```

- 结构化抽取：Pydantic `HealthFacts{facts}` + `with_structured_output`，从**本轮用户原话**
  （`state["question"]`，不是改写）抽健康事实。
- 防误抽：prompt 严限「只记明确陈述的过敏/慢病/用药，没有返回空」，不臆测。
- `store.put(ns, uuid, {"text": fact})`：写进该用户 namespace，key 用随机 uuid。
- try/except：失败跳过不影响作答。放图最后（answer 之后），是"答完顺手记一笔"。

## 一条数据怎么流

```
你说："我对青霉素过敏，嗓子发炎吃什么药"
  contextualize：首轮无历史 → standalone = 原问题
  recall_memory：档案空 → user_memory = []
  ...检索作答...
  answer：return history（这轮问答存短期）
  extract_memory：抽出"对青霉素过敏" → store.put 进档案 ✓

【新会话，同 user_id】你说："我感冒能吃头孢吗"
  recall_memory：search 命中"对青霉素过敏" → user_memory=["对青霉素过敏"]
  answer：注入档案 → "你对青霉素过敏，头孢可能交叉过敏，需谨慎..." ✓
```

answer 用 user_memory：
```python
mem = state.get("user_memory") or []
memory_block = "用户健康背景：\n" + "\n".join(f"- {m}" for m in mem) if mem else ""
# 拼进 prompt，并要求"有安全冲突主动提示"
```

## 面试问题清单

1. 短期和长期记忆代码上有什么区别？→ 短期隐式（动 state，checkpointer+reducer+thread_id 自动）；长期显式（自己 store.put/search + user_id namespace）。
2. history 怎么跨轮累积？→ `Annotated[list, operator.add]` reducer 自动追加 + checkpointer 持久化。
3. 节点怎么拿到 store？→ 签名声明 `store`，运行时注入。
4. 怎么保证 A 用户记忆不串到 B？→ user_id 做 namespace 隔离（`_user_ns`）。
5. 长期记忆怎么按相关性召回？→ store 配 BGE index，`search(query=)` 语义检索。
6. 为什么 extract 放最后、用原话不用改写？→ 不影响当前作答；原话才有用户完整陈述。
7. 防误抽怎么做？→ 结构化输出 + prompt 严限 + try/except。

## 自测题

1. `contextualize` 里 `state["history"]` 的内容是谁、什么时候写进去的？
2. 短期记忆代码里为什么看不到 thread_id，长期却要显式用 user_id？
3. `recall_memory` 和 `extract_memory` 的 `store` 参数怎么来的？
4. 新会话（换 thread_id、user_id 不变）后，三个节点各自还记得什么？

---

# 附：历史窗口（不是"只读 4 轮"）+ reducer 是什么

## 历史：存储全量，使用只看最近 4 轮

| | 实际情况 |
|--|--|
| 存储（checkpointer 里的 history）| **全量累积**，一条不丢 |
| 使用（`contextualize` 指代消解）| 只取**最近 4 轮**喂 LLM |
| 其他节点（如 answer）| **根本不读 history** |

截断就这一行：
```python
recent = history[-2 * HISTORY_WINDOW:]   # 最近 4 轮（×2 因一轮含 user+assistant 两条）
```
- 为什么只取 4 轮：指代（「它」）通常指最近提到的，取太多费 token、引噪声、变慢。
- 「超出的不计入」准确说：超出的**还在存储里**，只是指代消解时**不参考**。第 1 轮提的药到第 10 轮
  用「它」指，可能消解不到——这是 `HISTORY_WINDOW=4` 的局限。
- 能改：`config.py` 的 `HISTORY_WINDOW = 4`，调大参考更多轮（更费 token）。

## reducer = 字段「新旧值怎么合并」的规则

LangGraph 里每个节点 return 部分更新，要合并进总 state。**默认规则是覆盖**（新值替换旧值）。
reducer 用来改这个规则。

history 举例：

**默认覆盖（没 reducer）：**
```
旧: history=[A,B]  + answer return history=[C,D]  → [C,D]   旧的没了！多轮就废
```

**add reducer（`Annotated[list, operator.add]`）：**
```
旧: history=[A,B]  + answer return history=[C,D]  → [A,B,C,D]   拼接，历史累积
```
`operator.add` 对列表就是 `+`（拼接）。所以 `answer` 只 return 新增两条，历史却能一直累积。

```python
# med_state.py
history: Annotated[list[dict], operator.add]   # 累加：每轮追加，不覆盖
```

对比：

| 字段 | reducer | 合并方式 | 为什么 |
|------|---------|---------|--------|
| `question` | 无 | 覆盖 | 每轮问题换新的 ✓ |
| `standalone_question` | 无 | 覆盖 | 每轮重算 ✓ |
| `history` | `operator.add` | 累加 | 对话要积累 ✓ |

默认覆盖对大多数字段是对的（每轮重算）；只有要积累的（history）才需要 add reducer。

---

# 自测题参考答案

**1. `contextualize` 里 `state["history"]` 的内容是谁、什么时候写进去的？**

内容是历轮的「用户问 + 助手答」记录（`[{role:"user",...}, {role:"assistant",...}, ...]`）。
写入时机：每轮 `answer` 节点结尾 `return {"history": [{用户问}, {助手答}]}`；靠 `history` 字段的
`operator.add` reducer 追加、checkpointer 按 thread_id 持久化。下一轮同 thread_id 进来，
checkpointer 自动恢复，`contextualize` 才读得到。

**2. 短期记忆代码里为什么看不到 thread_id，长期却要显式用 user_id？**

短期记忆由 checkpointer 在背后**按 thread_id 自动存取整个 state**，节点只管读写 `state["history"]`，
不需要也看不到 thread_id（LangGraph 从 invoke 的 config 拿 thread_id 自动处理）。长期记忆是你
**自己调 `store.put/search`**，必须自己从 config 取 user_id 拼 namespace 来隔离用户，所以显式出现。
本质：**短期是框架托管，长期是手动操作。**

**3. `recall_memory` 和 `extract_memory` 的 `store` 参数怎么来的？**

节点签名里声明 `*, store: BaseStore`，LangGraph **运行时自动注入**——就是 `med_graph.py` 里
`compile(store=...)` 传进去的那个 `InMemoryStore`。`config` 同理也是自动注入。

**4. 新会话（换 thread_id、user_id 不变）后，三个节点各自还记得什么？**

- `contextualize`：换了 thread_id = 新的空 history，**记不得**上个会话的对话；首轮无历史，原样返回。
- `recall_memory`：user_id 没变 = 同一 namespace，**仍能召回**之前存的健康档案。
- `extract_memory`：写进同一 user_id namespace，新会话抽取的事实**追加进同一份档案**（与旧档案累积）。

一句话：**丢短期（对话历史），留长期（用户档案）。**
