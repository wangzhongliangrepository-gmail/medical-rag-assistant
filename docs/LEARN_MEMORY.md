# 学习串讲：Memory 三节点（短期读 / 长期读 / 长期写）

> 配合 `med_nodes.py`、`med_state.py`、`med_graph.py` 阅读。前置：thread_id vs user_id 隔离原理。

## 三节点定位

```
contextualize → recall_memory → plan → ... → answer → ... → extract_memory → END
  [短期·读历史]   [长期·读档案]                            [写短期+长期·写档案]
```

| 节点 | 属于 | 干什么 |
|------|------|--------|
| `contextualize` | 短期记忆 | 读对话历史+会话实体清单，把「它」改成具体实体（指代消解），并识别本轮药/病累积进清单 |
| `recall_memory` | 长期记忆 | 按 user_id 调出用户健康档案 |
| `extract_memory` | 短期+长期记忆 | 收尾：把本轮最终问答写进 history（短期）+ 抽用户健康信息存档案（长期）|

短期记忆 history 的「写」不在 `answer`，而**统一在收尾节点 `extract_memory`** 写一次（见下文节点 3 说明）。

## ⭐ 题眼：短期和长期，代码写法完全不同

| | 短期记忆（history） | 长期记忆（健康档案） |
|--|--|--|
| 钥匙 | thread_id | user_id |
| 存储 | checkpointer | store |
| 代码怎么操作 | **隐式**——只管读写 `state["history"]`，LangGraph 按 thread_id **自动**存取整个 state | **显式**——自己调 `store.search()`/`store.put()`，自己用 user_id 做 namespace |
| 看得到钥匙吗 | 看不到 thread_id（背后自动） | 看得到 user_id（自己从 config 取）|

一句话：**短期记忆是 LangGraph 替你管的（你只动 state），长期记忆是你自己管的（自己 put/search）。**

## 节点 1：contextualize（短期·读历史做指代消解 + 实体识别）

> ⚠️ 这个节点升级过：除了用历史做指代消解，还加了「**会话实体清单**」机制——按药/病类型**过滤** +
> **最近优先**地给 LLM 候选实体，解决「第 1 轮提的药、第 10 轮用『它』指代，但超出 4 轮历史窗口
> 够不着」的问题。一次结构化调用同时做两件事：① 消解 ②识别本轮药/病累积进清单。

```python
class Contextualized(BaseModel):       # 结构化输出：消解结果 + 本轮实体
    standalone_question: str
    drugs: list[str] = []
    diseases: list[str] = []

def contextualize(state, config):
    question = state["question"]
    history = state.get("history", [])
    entities = state.get("mentioned_entities", [])     # ① 会话累积的实体清单（最近的在尾）

    # ② 仅当含指代时才需要候选实体提示：先猜类型，再"过滤同类 + 最近优先"
    ref_type = _guess_ref_type(question) if _has_pronoun(question) else ""
    hint = _filter_and_order_entities(entities, ref_type) if _has_pronoun(question) else []
    entities_text = "\n".join(f"- {e['name']}（{'药' if e['type']=='drug' else '疾病'}）"
                              for e in hint) or "（无）"

    recent = history[-2 * HISTORY_WINDOW:]             # ③ 历史也只取最近几轮
    history_text = "\n".join(f"{h['role']}：{h['content']}" for h in recent) or "（无）"

    structured = _llm(config).with_structured_output(Contextualized, method="json_mode")
    result = structured.invoke([("system", "你是指代消解+实体识别器，只输出合法 JSON..."),
                                _CONTEXTUALIZE_PROMPT.format(history=history_text,
                                                             entities=entities_text, question=question)])
    new_entities = ([{"name": d, "type": "drug"} for d in result.drugs]
                    + [{"name": s, "type": "disease"} for s in result.diseases])
    return {"standalone_question": result.standalone_question.strip() or question,
            "mentioned_entities": new_entities}        # ④ 消解结果 + 累积本轮实体
```

两个核心 helper（指代消解的灵魂，`med_nodes.py`）：

```python
def _guess_ref_type(text):          # 据属性词猜「它」指什么类型 → 过滤依据
    # "副作用/禁忌/剂量"→drug；"症状/并发症/怎么治"→disease；都中或都不中→""（交 LLM）
    ...

def _filter_and_order_entities(entities, ref_type, limit=ENTITY_HINT_K):
    # ① 过滤：判出类型就只留同类（"它的副作用"→只看药，排除疾病）
    cands = [e for e in entities if e["type"] == ref_type] if ref_type else list(entities)
    # ② 最近优先：从尾（最近）往头去重，最近一次提及的排最前
    seen, ordered = set(), []
    for e in reversed(cands):
        if e["name"] and e["name"] not in seen:
            seen.add(e["name"]); ordered.append(e)
    return ordered[:limit]
```

- ① **实体清单 `mentioned_entities`**：`med_state.py` 里 `Annotated[list, operator.add]` 累加式，
  每轮 `contextualize` 把识别出的药/病追加进去，最近的在尾部。
- ② **过滤 + 最近优先**（增强的两个功能）：`_guess_ref_type` 用问题里的属性词（"副作用"）判出「它」
  指药还是病 → `_filter_and_order_entities` 先**只留同类**、再**最近优先排序**，把候选喂给 LLM 提示
  「优先用最靠前（最近）的同类实体替换指代」。
- ③ 历史仍 `[-2*HISTORY_WINDOW:]` 只取最近 4 轮（一轮 ×2 条）。
- ④ 输出 `standalone_question`（下游全用它）+ 累积本轮实体（供**后续轮**消解）。

> 💡 为什么实体清单能跨越 4 轮窗口：历史只看最近 4 轮，但实体清单是**全会话累积**的——
> 第 1 轮的"二甲双胍"即使早已滑出历史窗口，仍躺在清单里，第 10 轮"它"还能靠清单指回去。

**⚠️ 一个行为变化（面试可能问）**：升级后 `contextualize` **每轮都调一次 LLM**（原来首轮/自包含
问题直接 `return` 跳过）——因为要识别本轮实体播种清单。代价是首轮多一次轻量结构化调用，
换来跨历史窗口的指代能力。

**history 怎么写进去（注意：不在 `answer`）**：本轮最终问答由**收尾节点 `extract_memory`** 统一
`return {"history": [{user...}, {assistant...}]}` 写一次；`med_state.py` 里
`history: Annotated[list, operator.add]` 累加 reducer 自动追加，checkpointer 持久化。
> 为什么不在 `answer` 写？因为反思回路会让 `answer` **多次执行**，在 answer 写会把同一轮的问和
> 中间草稿重复记进 history。统一在收尾节点写，保证"一轮 = 一组最终问答"。详见节点 3。

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

## 节点 3：extract_memory（收尾：写短期 history + 写长期档案）

> 这是图的收尾节点，干**两件事**：① 把本轮最终问答写进**短期** history；② 抽健康事实写**长期**档案。

```python
def extract_memory(state, config, *, store):
    question = state.get("standalone_question") or state["question"]
    # ① 短期记忆：一轮只记一组最终问答（无论反思重答几次）
    hist_update = {"history": [{"role": "user", "content": question},
                               {"role": "assistant", "content": state.get("answer", "")}]}

    # ② 长期记忆：抽健康事实（失败不影响 history 写入）
    structured = _llm(config).with_structured_output(HealthFacts, method="json_mode")
    try:
        result = structured.invoke([("system", "你是健康信息抽取器..."),
                                    _EXTRACT_PROMPT.format(utterance=state["question"])])
    except Exception:
        return hist_update                           # 抽取失败也要把 history 写回
    ns = _user_ns(config)
    for fact in result.facts:
        if fact.strip():
            store.put(ns, str(uuid.uuid4()), {"text": fact})   # 逐条写入该用户档案
    return hist_update
```

- **① 写短期 history**：返回本轮 `(问, 答)` 两条，靠 `operator.add` reducer 累加进 history。
  **为什么 history 在这里写、不在 `answer` 写**：反思回路会让 `answer` 多次执行，在 answer 写会把
  同一轮的问和中间草稿重复记进 history；统一在收尾写，保证"一轮 = 一组最终问答"。
- **② 写长期档案**：Pydantic `HealthFacts{facts}` + `with_structured_output`，从**本轮用户原话**
  （`state["question"]`，不是改写）抽健康事实。防误抽：prompt 严限「只记明确陈述的过敏/慢病/用药，
  没有返回空」，不臆测。`store.put(ns, uuid, {"text": fact})` 写进该用户 namespace。
- try/except：抽取失败也照样 `return hist_update`——长期抽取出错不能连累短期 history 丢失。

## 一条数据怎么流

```
你说："我对青霉素过敏，嗓子发炎吃什么药"
  contextualize：首轮无历史 → standalone = 原问题；识别实体"咽炎/青霉素"入清单
  recall_memory：档案空 → user_memory = []
  ...检索作答...
  extract_memory：① 写 history（这轮问答存短期）② 抽出"对青霉素过敏" → store.put 进档案 ✓

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
8. history 在哪个节点写？为什么？→ 收尾节点 `extract_memory` 统一写一次；不在 `answer` 写是因为
   反思回路会让 answer 多次执行，会把同一轮的问和中间草稿重复记进 history。
9. 指代消解超出历史窗口怎么办？→ 维护「会话实体清单」`mentioned_entities`（全会话累积），消解时
   按「过滤同类（`_guess_ref_type` 据属性词判药/病）+ 最近优先（`_filter_and_order_entities` 倒序去重）」
   把候选实体喂给 LLM，让"它"能指回早已滑出 4 轮窗口的实体。
10. 多个候选实体时"它"指哪个？→ 默认"最近优先"，但同类过滤先砍掉不相干类型；仍歧义可进一步做
    反问澄清（本项目当前到"过滤+最近优先"为止，反问是可扩展方向）。

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
  用「它」指，靠纯历史窗口可能消解不到——这是 `HISTORY_WINDOW=4` 的局限。
- 能改：`config.py` 的 `HISTORY_WINDOW = 4`，调大参考更多轮（更费 token）。
- **本项目已部分缓解这个局限**：`contextualize` 额外维护「会话实体清单」`mentioned_entities`（全会话
  累积、不受 4 轮窗口限制），指代消解时按「过滤同类 + 最近优先」把候选实体喂给 LLM。所以即便
  "二甲双胍"早滑出历史窗口，仍在清单里、第 10 轮"它"还能指回去（见节点 1）。

## reducer = 字段「新旧值怎么合并」的规则

LangGraph 里每个节点 return 部分更新，要合并进总 state。**默认规则是覆盖**（新值替换旧值）。
reducer 用来改这个规则。

history 举例：

**默认覆盖（没 reducer）：**
```
旧: history=[A,B]  + extract_memory return history=[C,D]  → [C,D]   旧的没了！多轮就废
```

**add reducer（`Annotated[list, operator.add]`）：**
```
旧: history=[A,B]  + extract_memory return history=[C,D]  → [A,B,C,D]   拼接，历史累积
```
`operator.add` 对列表就是 `+`（拼接）。所以 `extract_memory` 只 return 新增两条，历史却能一直累积。
（`mentioned_entities` 也用同一个 `operator.add` reducer，所以会话实体清单同样是累积式。）

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
| `mentioned_entities` | `operator.add` | 累加 | 会话实体清单要积累（供指代消解）✓ |

默认覆盖对大多数字段是对的（每轮重算）；只有要积累的（history / mentioned_entities）才需要 add reducer。

---

# 自测题参考答案

**1. `contextualize` 里 `state["history"]` 的内容是谁、什么时候写进去的？**

内容是历轮的「用户问 + 助手答」记录（`[{role:"user",...}, {role:"assistant",...}, ...]`）。
写入时机：每轮**收尾节点 `extract_memory`** `return {"history": [{用户问}, {助手答}]}`（不在 `answer`
写——反思会让 answer 多次执行，统一收尾写防重复）；靠 `history` 字段的 `operator.add` reducer
追加、checkpointer 按 thread_id 持久化。下一轮同 thread_id 进来，checkpointer 自动恢复，
`contextualize` 才读得到。

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
