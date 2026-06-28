# 面试问答（Agent 基础 + 医疗 RAG 项目核心）

> 一份**问答驱动**的面试备战文档：左边问题、右边答题要点 + 「能多说一句」的加分项。
> 与 `INTERVIEW.md`（速查合订本）、各 `LEARN_*.md`（深度串讲）互补——本篇专门按
> 「面试官会怎么问、我该怎么答」组织，背完能直接对着代码讲。
>
> 用法：先看问题自己默答，再对答案；标 ★ 的是高频核心，标 ⚠️ 的是容易被追问翻车点。

---

## 目录

- [第一部分　Agent 基础重点](#第一部分agent-基础重点)
  - A. 概念与边界
  - B. 四大支柱
  - C. 主流范式
  - D. LangGraph
  - E. 工具调用与结构化输出
- [第二部分　项目核心](#第二部分项目核心)
  - F. 项目总览与数据流
  - G. Planning（规划拆解）
  - H. Tool Use（混合检索 + 知识融合）★核心
  - I. Reflection（反思回路）
  - J. Memory（短期 + 长期记忆）
  - K. 防幻觉与引用（医疗安全）
  - L. 工程与部署
  - M. 评测、短板与改进
- [第三部分　30 秒开场白](#第三部分30-秒开场白)

---

# 第一部分　Agent 基础重点

## A. 概念与边界

**A1. ★ 什么是 Agent？和普通 LLM 调用有什么区别？**
普通 LLM 调用是「prompt → 文本」一锤子买卖；Agent 让 LLM 成为**决策中枢**，能自主规划多步、
调用工具、保留记忆、反思纠错地完成任务。
> 一句话公式：**Agent = LLM（大脑）+ 工具（手脚）+ 记忆（上下文）+ 循环（反思纠错）**。

**A2. ★ Agent vs 普通 RAG？**
普通 RAG 是「检索一次 → 作答」的**固定单向流程**；Agent 能先**规划**多步、调**多个**工具、
答完还能**反思**够不够再补检索重试，并带**记忆**跨轮。
> 加分：本项目其实是「RAG 长成了 Agent」——基线是纯 RAG，逐层加了 Planning / 知识融合 /
> Reflection / Memory，最后四大支柱齐全。

**A3. Agent vs Workflow（工作流）？**
Workflow 是**人预先编排好的固定路径**；Agent 让 **LLM 自主决策走哪条路**。实践中常混合。
> 本项目偏「**LLM 决策 + 图约束**」：拆解/反思判断/记忆抽取都由 LLM 定，但整体路径由
> LangGraph 图骨架约束——可控性和自主性的折中，工程上更稳。

**A4. ⚠️ 你这个项目算「真 Agent」吗？工具调用是 LLM 自己决定的吗？**
诚实答：**检索工具是图里固定编排调用的**（更接近 workflow 式工具使用），不是严格的
function calling 让 LLM 自选工具。但**反思回路**里「要不要再检索、补检索查什么」是 LLM
决策的，外部检索还有 `use_external` 用户开关。
> 这么答显诚实又懂边界。可扩展方向：把检索改成 LLM 自主 function calling。

---

## B. 四大支柱

**B1. ★ Agent 四大支柱是什么？分别对应你项目哪里？**

| 支柱 | 是什么 | 本项目落地 |
|------|--------|-----------|
| **Planning** | 把复杂任务拆成子步骤 | `plan` 节点把复合问诊**平行拆解**成 1-3 个方面子问题 |
| **Tool Use** | 调工具拿自身没有的信息 | 内部 Qdrant 混合检索 + 外部 Tavily 联网，两路 `fuse` 融合 |
| **Reflection** | 自我批判、不够重来 | `reflect` 答完自判证据是否充分，不足 → `augment_retrieve` 补检索重答（带上限） |
| **Memory** | 跨轮/跨会话保留上下文 | 短期 `InMemorySaver`（指代消解）+ 长期 `InMemoryStore`（BGE 语义召回用户健康事实） |

> 一句话：**四大支柱全部落地**。这是本项目最大的卖点，开场就抛。

**B2. 四大支柱里哪个最难？为什么？**
个人认为 **Reflection + Memory** 工程上最难：Reflection 要处理**循环 + 终止条件**（防死循环）
和「证据累积不覆盖」；Memory 要分清**短期隐式（框架托管）vs 长期显式（自己 put/search）**、
做好 user 隔离和防误抽。Tool Use 技术含量最高但模式成熟（混合检索是工业标准做法）。

---

## C. 主流范式

**C1. ★ ReAct vs Plan-and-Execute vs Reflexion？你用了哪些？**

- **ReAct**：Thought → Action → Observation 循环，边想边做。灵活，但慢、贵、易跑偏。
- **Plan-and-Execute**：先一次性规划再执行。可控、调用少，但不够灵活。
  → 本项目 Planning 用这个，且是**平行拆解**（子问题互相独立、可同时检索）。
- **Reflexion**：执行后自我反思 → 发现不足 → 补救重试，带修订次数上限。
  → 本项目 Reflection 用这个。

**C2. 为什么 Planning 选 Plan-and-Execute 而不是 ReAct？**
医疗问诊的复合问题多是「多个并列方面」（机理 / 用药 / 副作用），适合**一次性平行拆解**
分别检索，可控、调用少、好缓存；ReAct 边想边做对这种结构是浪费，还更易跑偏。

---

## D. LangGraph

**D1. ★ LangGraph 和 LangChain 的区别？**
LangChain 是「**积木**」（LLM 封装、检索器、解析器 + LCEL 线性链），擅长**线性/DAG**；
LangGraph 是「把积木按**带循环的流程图**组装起来的引擎」，把应用建模成**状态图**，专门处理
**有状态、多步、会循环、要分支**的 Agent。
> 关系：同生态、互补。**LangGraph 编排控制流，节点内部用 LangChain 组件**（如
> `ChatDeepSeek`、`with_structured_output`）。

**D2. ★ 为什么这个项目必须用 LangGraph，不用 LangChain 链？**
三样东西纯 LCEL 链做不优雅：
1. **反思回路是循环**：`answer → reflect →(不足)→ augment_retrieve → fuse → answer`，带终止条件的环；
2. **条件分支**：reflect 后「充分 → 收尾 / 不足 → 补检索」靠条件边；
3. **持久化状态/记忆**：多轮靠 checkpointer 按 thread_id 存取 state，长期记忆靠 Store。
> 一句话：**因为我的 Agent 要循环、要分支、要跨轮记忆——这正是 LangGraph 相对链的核心价值。**

**D3. ★ LangGraph 核心概念对照你的代码？**

| 概念 | 是什么 | 本项目 |
|------|--------|--------|
| StateGraph | 状态图本体 | `g = StateGraph(MedState)` |
| State | 节点间共享数据 | `MedState`（TypedDict） |
| Reducer | 字段怎么合并（覆盖/累加） | `history: Annotated[list, operator.add]`（累加） |
| Node | `(state)→部分state` 的函数 | `plan`/`retrieve_*`/`answer`… |
| Edge | 固定连接 A→B | `add_edge("plan","retrieve_internal")` |
| Conditional Edge | 按 state 决定走向 | `add_conditional_edges("reflect", route_after_reflect, {...})` |
| compile | 编译成可执行图 + 挂记忆 | `g.compile(checkpointer=, store=)` |

**D4. ⚠️ reducer 是什么？不写会怎样？**
reducer 决定一个 state 字段「新旧值怎么合并」。**默认是覆盖**（新值替换旧值）。
`history` 必须用 `operator.add`（拼接累加），否则每轮 return 新两条会**覆盖**掉旧历史，多轮就废了。
其余字段（如 `question`/`standalone_question`）每轮重算，用默认覆盖即可。

**D5. 循环怎么形成，又怎么不死循环？**
循环：`augment_retrieve → fuse` 这条边**指回上游已走过的 fuse**，兜成环（这正是 LangGraph
区别于 LangChain 链的核心——能往回连）。防死循环：路由函数 `route_after_reflect` 判断
`revisions >= MAX_REVISIONS`（=2），到上限强制走 `extract_memory` 断环。

**D6. 节点怎么拿到 store / config？**
**依赖注入**：节点签名声明 `config: RunnableConfig`、`*, store: BaseStore`，LangGraph 运行时
自动注入（store 就是 `compile(store=...)` 传进去那个）。运行时配置经
`invoke(..., config={"configurable": {"thread_id":..., "user_id":..., "model_tier":...}})` 传入。

**D7. 还没用但该知道的 LangGraph 边界（防被问倒）？**
streaming（`graph.stream()`）、human-in-the-loop / interrupt（中途暂停等人工确认）、
subgraph（子图复用）、持久化后端（`SqliteSaver`/`PostgresSaver` 替内存版）。
> 答「知道，本项目没用上」即可——显示懂边界。

---

## E. 工具调用与结构化输出

**E1. Function Calling 怎么工作？**
LLM 输出**结构化的工具调用意图（JSON）**，框架执行真实函数，把结果回喂给 LLM。

**E2. ★ 为什么用结构化输出而不是正则抠 JSON？**
正则解析自由文本脆弱易碎。本项目所有「让 LLM 吐结构」的地方都用
`with_structured_output(PydanticModel, method="json_mode")`——DeepSeek 直接返回合法 JSON，
Pydantic 强校验。用在：`plan`（SubQuestions）、`reflect`（ReflectResult）、`extract_memory`（HealthFacts）。

**E3. 工具失败 / LLM 返回空怎么办？**
- 外部 API 失败：`try/except` **优雅降级**（Tavily 挂了返回空、仅用内部源，服务不崩）。
- LLM 返回空：**兜底**（如 plan 的 `subs = result.questions or [question]` 退回原问题）。
- 结构化抽取失败：try/except 跳过，不影响主流程作答。

---

# 第二部分　项目核心

## F. 项目总览与数据流

**F1. ★ 一句话介绍这个项目？**
一个带引用、内外部知识融合、会反思、有记忆的**中文医疗知识助手（RAG + Agent）**：用户问一句
医疗问题 → 规划拆解 → 在「内部医学教材库（Qdrant 混合检索）+ 外部 Web（Tavily）」两路找证据 →
融合重排 → DeepSeek 带 `[编号]` 引用、严格防幻觉作答 → 反思证据够不够 → 抽取用户健康事实入长期记忆。
LangGraph 编排，FastAPI + 网页前端 + Docker 部署。

**F2. ★ 完整数据流（一个问题走过的路）？**
```
question
 → contextualize   用 history 做指代消解 → standalone_question      [短期记忆·读]
 → recall_memory   按 user_id 从 Store 语义召回用户健康事实         [长期记忆·读]
 → plan            平行拆成 1-3 个子问题                            [Planning]
 → retrieve_internal  对每个子问题 Qdrant 混合检索（去重累积）       [Tool Use]
 → retrieve_external  use_external 开则 Tavily 联网（失败降级）      [Tool Use]
 → fuse            内外部证据合并 → 对原问题统一重排 top-k
 → answer          只依据证据、带 [编号] 引用、结合健康背景安全提示
 → reflect         证据够不够？                                     [Reflection]
     ├─ 不足   → augment_retrieve（用 missing_info 补检索）→ fuse → answer（≤MAX_REVISIONS）
     └─ 充分/达上限 → extract_memory（抽健康事实写回 + 记本轮问答）  [长期+短期记忆·写] → END
```

**F3. 项目怎么分层组织的？**
**两层、关注点分离**：
- **基础设施层**（连服务、干脏活）：`llm.py` / `embeddings.py` / `kb_search.py` / `sparse.py` /
  `vectordb.py` / `external.py` / `ingest.py` / `config.py`。
- **编排层**（串流程）：`med_state.py`（状态）/ `med_nodes.py`（节点）/ `med_graph.py`（连图）/
  `server.py`·`med_rag.py`（入口）。
> 好处：改检索算法不动流程，改流程不动检索。

---

## G. Planning（规划拆解）

**G1. ★ 为什么要拆解问题，不拆直接检索不行吗？**
**单次检索覆盖不全。** 例：「二甲双胍的副作用和禁忌症」不拆，向量检索可能主要召回「副作用」段落，
「禁忌症」被挤掉 → 答案只讲副作用。拆成两个子问题分别检索，两方面证据都进池，答案才完整。
> 因为下游 `retrieve_internal` 是**逐子问题分别检索再累积去重**的。

**G2. ⚠️ 拆解一定更好吗？平行拆解有什么局限？**
不一定。**过度拆解打散语义、引噪声、白花检索**，所以 prompt 明说「单一问题只输出一个子问题」。
更关键的局限：**平行拆解搞不定「桥接型/多跳」问题**。例：「治高血压的一线药，它的副作用是什么」——
要先查「一线药是谁（如 ACEI）」才能查「它的副作用」。平行拆会拆出悬空的「那个药的副作用」，
检索全是噪声。解法是**链式精化**（先检索第一跳、代入第二跳再检索），设计见 `docs/P2_5_DESIGN.md`，
当前 `plan` 是平行版。
> 能说出局限 + 知道怎么解 = 满分答案。

**G3. 怎么保证 LLM 拆解输出可解析？拆失败怎么办？**
结构化输出 `with_structured_output(SubQuestions, method="json_mode")` + Pydantic，不用正则；
返回空时 `or [question]` 兜底原问题。

**G4. 拆的是原问题还是改写后的？**
拆的是 `standalone_question`（指代消解后的）。所以「那它的禁忌呢」已先被改成「二甲双胍的禁忌」
再拆，子问题里不含悬空指代——**Planning 衔接 Memory**。

---

## H. Tool Use（混合检索 + 知识融合）★核心

**H1. ★ 为什么用混合检索，不直接向量检索？**
**dense 抓语义、sparse 抓精确词面，互补。**
- 纯 dense（语义向量）：问「二甲双胍」可能混回一堆别的降糖药，精确性不足。
- 纯 sparse（BM25 词面）：问「降血糖的药」对不上「二甲双胍」，召回不到同义改写。
- 医疗里药名要精确（sparse 强），也要懂同义口语（dense 强）→ 两路都要。

**H2. ★ dense 和 sparse 具体是什么？**

| | dense（稠密/语义） | sparse（稀疏/词面） |
|--|--|--|
| 本质 | bge-m3 **神经网络** embedding | BM25 **统计算法**（非模型） |
| 在哪算 | Xinference（GPU） | FastEmbed 本地（CPU） |
| 向量 | 1024 维都有值 | 几万词表里只几个非零 |
| 抓什么 | 语义相近 | 精确词面 + 词频 |
| 懂同义 | 懂 | 不懂 |

> ⚠️ 加分点：严格说**只有 bge-m3 是真正的「向量化模型」，BM25 是统计打分算法**——只是在 Qdrant
> 混合框架里它的结果也被表示成「稀疏向量」，所以统称两路向量。

**H3. ★ RRF 是什么，为什么用它？**
**Reciprocal Rank Fusion（倒数排名融合）**：两路分数尺度不同（dense 是 cosine 0~1，sparse 是
BM25 几十），没法直接比。RRF **不看分数、看排名**，每条最终分 = `Σ 1/(k + 该路排名)`（k=60）。
好处：**免去归一化不同尺度，简单鲁棒**；两路都靠前的浮上来。Qdrant 原生支持（`FusionQuery(fusion=RRF)`）。

**H4. ★ 召回和重排为什么分两步（两阶段架构）？**
精排模型（cross-encoder）准但慢，对 5 万+ 段逐个打分跑不动。所以：
1. **召回 recall**（RECALL_K=20）：快方法粗筛广撒网，从 5 万捞 20 条候选。
2. **重排 rerank**（RERANK_TOP_K=3）：慢但准的 cross-encoder 精排这 20 条取 top-3 喂 LLM。
> 兼顾速度和精度，工业级 RAG 标准做法。

**H5. ⚠️ 召回的双塔和重排的 cross-encoder 区别？**
- 召回 dense 用**双塔 bi-encoder**：query 和 doc **分别**编码再比向量距离——快（doc 向量可预存），但粗（没看交互）。
- 重排用 **cross-encoder**：query 和 doc **拼一起**进模型出相关度——准（逐词看对应关系），但慢（每条现算，没法预存）。
> 「能不能预存」是召回快、重排慢的根本原因，也是 cross-encoder 只用于已缩到 20 条之后的原因。

**H6. 召回为什么快？**
两招：① **向量预存**——dense 向量灌库时就算好存 Qdrant，查询只算 1 个 query 向量；
② **索引**——dense 路用 HNSW（近似最近邻，O(N)→约 O(log N)），sparse 路用倒排索引，都不暴力扫全库。

**H7. 知识融合（fuse）做了什么？**
把内部教材证据 + 外部 Web 证据**混在一起，对原问题统一重排取 top-k**——最相关的浮上来，
不论来源，而非机械各取几条。内外冲突时在作答里提示分歧。

**H8. 为什么 embedding/rerank 用 Xinference 不用 DeepSeek？**
**DeepSeek 没有 embedding 接口**，向量化和重排留在 Xinference 的 BGE（本地 GPU）。
> 细节：`XinferenceEmbeddings` 传 model_uid（不是模型名）；rerank 没现成 LangChain 封装，打 `/v1/rerank` REST。

**H9. 灌库为什么一个 chunk 存两个向量？**
**存查对称**：检索时要用 dense + sparse 两路分别查再 RRF 融合，所以灌库时每段就同时算 dense + sparse
两个向量存进 Qdrant（命名向量 `dense` + `sparse`）。

**H10. chunk 大小怎么定？**
BGE max_tokens=512，中文约 1 字 ≈ 1 token，控制 400 字内最稳（太长截断丢信息，太短语义不完整）。

**H11. 外部检索（Tavily）的三个工程要点？**
① `use_external` **用户开关**（前端 🌐，默认关——更快更省）；② **优雅降级**（try/except，挂了仅用内部源）；
③ 网页正文**截断 800 字**（避免拖慢重排、对作答无必要）。

---

## I. Reflection（反思回路）

**I1. ★ Reflection 怎么实现的？**
`answer` 后进 `reflect` 节点，LLM 自判「当前证据能否支撑**安全、完整**的回答」（结构化输出
`{sufficient, reasoning, missing}`）。不足则给出**具体补充检索查询** `missing_info` →
`augment_retrieve` 补检索 → `fuse` 重排 → `answer` 重答；条件边 `route_after_reflect` 控制回路。

**I2. ⚠️ 怎么保证不死循环？**
两层：① 硬上限 `MAX_REVISIONS=2`，路由函数判 `revisions >= 上限`强制收尾；
② **reflect prompt 克制**——明确「不要因为还能更详尽就判 false，只有缺**关键**信息才判 false」，
否则模型会无限觉得「还能更好」。

**I3. ⚠️ 补检索的证据会覆盖原来的吗？**
**不会，是去重追加累积。** `augment_retrieve` 用 `missing_info` 检索后，按 text 去重**追加**到
`internal_evidence`/`external_evidence`，`fuse` 每轮对**全部**证据统一重排——证据只增不减。

**I4. ⚠️ missing_info 不是对原问题复述吧？怎么保证它有用？**
prompt 强约束：missing 必须是**含药名/方面的具体补充检索查询**，不能是「还有什么要补充」这种空话。
示例引导：✓「二甲双胍的禁忌症和用药注意」 ✗「还有什么要补充的」。

**I5. 怎么证明反思在起作用？**
实测「二甲双胍的副作用、禁忌、漏服处理」→ reflect 识别「漏服处理缺失」→ 补检索 → 证据增至 10 条 →
答案补全，revisions=2。`get_med_graph(use_reflect=False)` 可关反思做**消融对照**。

**I6. reflect 自己失败了怎么办？**
try/except，**视为充分**继续收尾（`missing_info=""`），不让反思异常拖垮主流程。

---

## J. Memory（短期 + 长期记忆）

**J1. ★ 短期和长期记忆怎么分？**

| | 短期记忆（对话 history） | 长期记忆（用户健康档案） |
|--|--|--|
| 撑什么 | 会话内多轮 | 跨会话 |
| 钥匙 | thread_id | user_id |
| 存储 | checkpointer（`InMemorySaver`） | store（`InMemoryStore`，BGE 语义索引） |
| 代码 | **隐式**——只读写 `state["history"]`，框架按 thread_id 自动存取 | **显式**——自己 `store.put/search` + user_id 做 namespace |

> ⭐ 题眼一句话：**短期是 LangGraph 替你管的（你只动 state），长期是你自己管的（自己 put/search）。**

**J2. ★ 多轮指代消解怎么做？**
`contextualize` 节点用最近 N 轮 history 把「那它的禁忌呢」改写成自包含的「二甲双胍的禁忌」
（standalone_question），下游 plan / 检索全用改写后的问题。首轮无历史则原样返回。

**J3. history 怎么跨轮累积？**
`med_state.py` 里 `history: Annotated[list, operator.add]`（累加 reducer）+ checkpointer 持久化。
节点只 return 新增两条，框架自动追加、按 thread_id 存取。

**J4. ⚠️ 短期 history 在哪个节点写入？**
**统一在收尾节点 `extract_memory` 写一次**（一轮只记一组最终问答）。
> ⚠️ 关键设计：**不在 `answer` 里写**——因为反思会让 answer **多次执行**，若在 answer 写会把
> 同一轮的问和中间草稿重复记进 history。统一在收尾写，保证「一轮 = 一组最终问答」。
> （注：早期 `LEARN_MEMORY.md` 描述的是 answer 写入，已被此设计取代，以代码为准。）

**J5. ★ 长期记忆怎么写、怎么读？**
- **写** `extract_memory`：用结构化输出从**本轮用户原话**抽取明确陈述的过敏史/慢病/用药，
  `store.put(ns, uuid, {"text": fact})`。
- **读** `recall_memory`：`store.search(ns, query=, limit=k)` 用 BGE **语义召回**相关健康事实，
  注入 answer 的 prompt 做安全提示。

**J6. ⚠️ 记忆怎么不误抽、不串用户？**
- 不误抽：抽取 prompt 严限「只记**明确陈述**的过敏/慢病/用药，没有返回空，不臆测、不把一次性症状当长期事实」+ try/except。
- 不串用户：**user_id 做 namespace 隔离**（`_user_ns`），A 查不到 B。

**J7. 长期记忆为什么用语义检索而不是全量塞进 prompt？**
用户健康事实可能很多，全塞费 token 且引噪声。store 配 BGE index，按当前问题
`search(query=)` 只召回最相关的 top-k——精准且省。

**J8. 为什么 extract 放最后、用原话不用改写？**
放最后是「答完顺手记一笔」，不影响当前作答；用**原话**（`state["question"]`）因为原话才有用户
完整陈述（改写后的 standalone_question 可能丢掉「我对青霉素过敏」这类背景）。

**J9. ⚠️ 重启后记忆还在吗？**
内存版（`InMemorySaver` + `InMemoryStore`）**进程级、重启清空**。生产可换 `SqliteSaver` +
Qdrant-backed Store 持久化——这是已知路线图，被问到大方承认。

---

## K. 防幻觉与引用（医疗安全）

**K1. ★ 医疗场景为什么特别需要 RAG + 引用？**
医疗容错率低、幻觉有害。RAG 把答案**锚定可信教材 + 实时 Web**，强制 `[编号]` 引用**可溯源核查**，
资料不足如实说明，对外一律附免责声明「仅供学习演示，非医疗建议」。

**K2. ★ 怎么防幻觉？**
四件事：① 系统/作答 prompt 严约束「**只依据资料、不编造、不足说不足、内外冲突指出分歧**」；
② 强制**关键结论后标 `[编号]`** 引用可溯源；③ 结构化输出约束格式；④ 无证据时直接返回
「根据现有资料无法回答」。

**K3. 内部教材 vs 外部 Web 各起什么作用？**
教材是**稳定基础知识**，Web 补**时效/最新指南/教材外内容**；冲突时提示分歧让用户判断。

**K4. 长期记忆怎么用于医疗安全？**
`recall_memory` 召回用户过敏史/慢病注入作答，prompt 要求「若建议药物与用户过敏史/禁忌冲突，
必须主动提示并给替代方向」。例：记得用户「青霉素过敏」，问「感冒能吃头孢吗」时主动提示交叉过敏风险。

---

## L. 工程与部署

**L1. ★ 本地和线上怎么零改动切换？**
**配置与代码解耦（12-Factor）**：所有端点/模型/密钥只从 `config.py` 取，`.env` 覆盖。
`QDRANT_URL` 一个变量——设了连服务器、不设走本地落盘，零代码改动切本地/部署。

**L2. ⚠️ `KMP_DUPLICATE_LIB_OK` 那个坑是什么？**
torch 和 onnxruntime 都带 OpenMP，重复加载会在 Windows 崩溃（OMP Error #15）。
`_bootstrap.py` 设 `KMP_DUPLICATE_LIB_OK=TRUE` 放行，且必须**在 import torch/onnx 之前**——
所以约定入口脚本第一行 `import _bootstrap`。

**L3. 动态选模型怎么做的，会丢记忆吗？**
节点内 `_llm(config)` 按请求的 `model_tier`（flash/pro）动态选模型并 `lru_cache` 复用；
checkpointer/store 是**编译时单例、所有请求共享**，所以切模型不丢记忆。默认 `deepseek-v4-flash`，
难节点可选 `deepseek-v4-pro`。

**L4. Docker 部署架构？**
docker-compose 编排 **app（FastAPI :8000）+ qdrant（:6333）** 在一个内部网络；
**Xinference 留宿主机（要 GPU）**，app 容器经 `host.docker.internal:9997` 连；
DeepSeek/Tavily 是云端 API，密钥运行时 `.env` 注入不烤进镜像。

**L5. Dockerfile 为什么先拷 requirements 再拷代码？**
**分层缓存**：requirements 不变就复用 pip 安装层（最慢的一层），改代码不重装包。
另外 `--host 0.0.0.0`（容器内 127.0.0.1 宿主访问不到）、Volume 持久化 qdrant 数据、
`depends_on` 只保证启动顺序不保证 ready（严格需 healthcheck）。

**L6. 上云的卡点？**
**GPU**：Xinference 要 GPU 普通云主机没有。三选一：① 租 GPU 实例；② 嵌入/重排换云端 API（最省钱）；
③ Qdrant Cloud + 云端嵌入 API，全程无 GPU。换 Docker / 上云后 qdrant 是空的需**重新灌库**
（本地落盘和容器 qdrant 是两套独立存储）。

**L7. FastAPI 服务怎么复用图？**
`server.py` 启动时**编译图一次**存全局变量 `GRAPH`，每个 `/chat` 请求复用同一个 GRAPH——
这也是记忆能跨请求生效的前提（同一套 checkpointer/store）。接口：`/`（前端）、`/health`、`/chat`。

---

## M. 评测、短板与改进

**M1. ★ 医疗答案怎么评估？为什么不用 EM/F1？**
医疗答复是**开放长文本**，EM/F1（精确匹配）失效。用两层：
① **检索 recall@k**（金标 chunk 是否被召回）——评检索；
② **LLM-as-judge**——评答案质量（忠实度/引用准确/完整性）。见 `docs/P1_DESIGN.md`、`HYBRID_RETRIEVAL_EVIDENCE.md`。

**M2. 怎么证明混合检索比纯向量好？**
`compare_retrieval.py` 做混合 vs 纯 dense 对比实证；证据见 `HYBRID_RETRIEVAL_EVIDENCE.md`。

**M3. ⚠️（主动说显诚实）项目有哪些短板 / 怎么改进？**
- **检索评测不够客观**：医疗线缺 HotpotQA 那样的客观 EM/F1，可补医疗 QA 测试集 + 召回率/引用准确率指标。
- **Planning 只支持平行拆解**：桥接型多跳问题要链式精化（`P2_5_DESIGN.md`）。
- **记忆是内存版**：重启清空，生产换 SqliteSaver + Qdrant-backed Store。
- **工具调用是固定编排**：非 LLM 自主 function calling，可扩展成自选工具。
- **防过时医疗信息**：靠来源标注 + 冲突提示 + 免责声明，可加来源可信度加权。
- **性能**：已做「编译图一次复用」，可加缓存（DeepSeek 上下文缓存）、qdrant 服务器版 HNSW。

---

# 第三部分　30 秒开场白

**Agent 能力版：**
> 我用 LangGraph 在一个医疗助手上落地了 Agent **四大支柱**：Planning 用 plan-and-execute 把复合
> 问诊平行拆成多方面；Tool Use 是 Qdrant 混合检索（dense+sparse RRF + BGE 重排）+ Tavily 联网两路工具、
> 再 fuse 融合；Reflection 用 reflect 节点答完自判证据是否充分、不足则用具体缺失查询补检索重答
> （带修订上限防死循环）；Memory 用 checkpointer 撑会话内多轮指代消解、用带 BGE 语义检索的 Store
> 跨会话记住用户过敏史/慢病并在作答时做安全提示。全程严格防幻觉、带 `[编号]` 引用可溯源。

**项目版：**
> 一个内外部知识融合的中文医疗 RAG：内部 Qdrant 做 dense+sparse 混合检索加 BGE 重排，外部 Tavily
> 联网补时效，两路证据融合后统一重排，DeepSeek 带引用作答严格防幻觉。LangGraph 编排
> plan→检索→融合→作答→反思→记忆，FastAPI + 网页前端，Docker compose 部署，靠环境变量实现
> 本地到容器零改动切换。

---

> 📌 配套深读：`LEARN_OVERVIEW`（全局地图）、`LEARN_LANGGRAPH` / `LEARN_PLANNING` /
> `LEARN_RETRIEVAL` / `LEARN_MEMORY`（四支柱深度串讲）、`LEARN_GLOSSARY`（英文术语读音）、
> `P*_DESIGN` / `P6_DEPLOY`（各阶段设计与部署）、`INTERVIEW`（速查合订本）。