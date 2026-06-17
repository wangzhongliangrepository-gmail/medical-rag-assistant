# 学习串讲：项目全局地图

> 第一讲。目标：脑子里有张整体的图——项目分几层、每个文件干嘛、一个问题怎么走完全程。

## 它到底是个什么东西

一句话：**一个带记忆、会反思的中文医疗问答助手**。用户问一句医疗问题，它去「内部医学
教材库 + 外部网络」找证据，结合用户健康背景，给出带引用、防幻觉的回答。

## 整个项目分两层（最重要的地图）

```
┌─────────────────────────────────────────────┐
│  第二层：Agent 编排层（LangGraph 把流程串起来）   │
│   med_state.py   状态（节点间传递的数据）          │
│   med_nodes.py   各个节点（每个节点干一件事）       │
│   med_graph.py   把节点连成一张图                  │
│   server.py / med_rag.py   对外入口（网页 / 命令行）│
└───────────────────┬─────────────────────────┘
                    │ 调用
┌───────────────────┴─────────────────────────┐
│  第一层：基础设施层（连外部服务，干脏活累活）       │
│   llm.py         连 DeepSeek（大模型）            │
│   embeddings.py  连 Xinference 的 BGE（向量化+重排）│
│   kb_search.py   在 Qdrant 里做混合检索           │
│   sparse.py      BM25 词面向量                    │
│   vectordb.py    连 Qdrant 数据库                 │
│   external.py    连 Tavily（联网搜索）            │
│   ingest.py      把教材灌进 Qdrant（一次性脚本）    │
│   config.py      所有配置的唯一入口               │
└─────────────────────────────────────────────┘
```

**为什么分两层？** 软件设计基本功——**关注点分离**。基础设施层只管「怎么连服务、怎么
检索」，编排层只管「先做什么后做什么」。好处：改检索算法不影响流程，改流程不动检索。
面试问「你项目怎么组织的」，这就是答案。

## 一个问题进来走过的路（数据流主干）

```
用户问「那它的禁忌呢？」
   │
contextualize  →  把「它」换成上文的「二甲双胍」（指代消解）   ← 记忆
   │
recall_memory  →  调出用户的健康背景（如「青霉素过敏」）       ← 记忆
   │
plan           →  把复合问题拆成几个方面                      ← Planning
   │
retrieve       →  内部教材库 + 外部网络 两路找证据             ← Tool Use
   │
fuse           →  两路证据合并、重排，取最相关的几条
   │
answer         →  结合证据+用户背景，带[引用]作答、安全提示
   │
reflect        →  自检：证据够不够？不够就回去补检索重答        ← Reflection
   │
extract_memory →  把这句话里的健康信息存进长期记忆            ← 记忆
   │
输出答案
```

## 四大支柱落在哪（面试核心）

| 支柱 | 对应节点 | 一句话 |
|------|---------|--------|
| **Planning** | `plan` | 复合问题拆成多方面分别检索 |
| **Tool Use** | `retrieve_internal/external` + `fuse` | 混合检索 + 联网 + 知识融合 |
| **Reflection** | `reflect` + `augment_retrieve` | 证据不够自己补检索重答 |
| **Memory** | `contextualize`/`recall_memory`/`extract_memory` | 短期多轮 + 长期用户记忆 |

## 核心就记两件事

1. **分两层**：基础设施层（连服务、检索） / 编排层（串流程）。
2. **一个问题走过那串节点**，每个节点对应一个支柱。

## 自测题

1. 「基础设施层」和「编排层」的区别，用自己的话说一遍。
2. 数据流里哪几个节点是「记忆」相关的？
3. 四大支柱分别对应哪些节点？

---

# 附：逐段讲解 `med_graph.py`（编排层核心）

这个文件本身不干活，只负责**把 `med_nodes.py` 里的节点拼成一张流程图**。节点是「工人」，
这个文件是「画流水线图纸」。

## 1. import 段

```python
from langgraph.checkpoint.memory import InMemorySaver   # 短期记忆
from langgraph.graph import END, START, StateGraph      # 建图三件套
from langgraph.store.memory import InMemoryStore        # 长期记忆
from embeddings import get_embeddings                   # BGE，给长期记忆做语义索引
from med_nodes import (answer, augment_retrieve, ...)   # 所有节点函数
from med_state import MedState                          # 状态结构
```

三类：① LangGraph 建图工具 + 记忆设施；② 自己写的节点；③ 状态结构。逻辑在 `med_nodes.py`，
编排在这里——关注点分离。

## 2. 函数签名 + 开关

```python
def get_med_graph(use_reflect: bool = True):
```

`use_reflect` 是**反思回路开关**：默认 True 走完整反思，False 退回「答完直接结束」。作用是
**对照实验**（消融对比"加反思 vs 不加"）。

## 3. 创建两套记忆设施

```python
checkpointer = InMemorySaver()           # 短期：按 thread_id 存会话 state
store = InMemoryStore(index={            # 长期：跨会话用户记忆
    "embed": get_embeddings(),           # 用 BGE 把记忆向量化
    "dims": EMBED_DIM,                   # 1024 维
    "fields": ["text"],                  # 对哪个字段建语义索引
})
```

`index=...` 让 store 支持**语义检索**（存进去的健康事实能按"意思相近"召回）。复用现成
`get_embeddings()`。它们在函数里创建 → 每调一次是一套新空记忆；`server.py` 只调一次，
所以整个服务进程共享同一套（"进程级单例"）。

## 4. 建图 + 注册节点

```python
g = StateGraph(MedState)              # 建一张以 MedState 为状态的空图
g.add_node("contextualize", contextualize)   # 名字 → 函数，只登记不连线
... # recall_memory / plan / retrieve_* / fuse / answer / extract_memory
```

注意 `add_node("plan", plan)` 直接传函数、**没绑 llm**——因为节点内用 `_llm(config)` 按请求
动态选 flash/pro。

## 5. 连固定边（主干流水线）

```python
g.add_edge(START, "contextualize")
g.add_edge("contextualize", "recall_memory")
g.add_edge("recall_memory", "plan")
g.add_edge("plan", "retrieve_internal")
g.add_edge("retrieve_internal", "retrieve_external")
g.add_edge("retrieve_external", "fuse")
g.add_edge("fuse", "answer")
```

`add_edge(A, B)` = **单行道**：A 跑完必去 B。`START` 是入口哨兵。

## 6. 反思回路（重点：循环在这里）

```python
if use_reflect:
    g.add_node("reflect", reflect)
    g.add_node("augment_retrieve", augment_retrieve)
    g.add_edge("answer", "reflect")                          # 答完去反思
    g.add_conditional_edges("reflect", route_after_reflect, {  # 条件分支（岔路口）
        "extract_memory": "extract_memory",     # 充分/达上限 → 收尾
        "augment_retrieve": "augment_retrieve", # 不足 → 补检索
    })
    g.add_edge("augment_retrieve", "fuse")      # ★ 补检索后回 fuse，形成环
else:
    g.add_edge("answer", "extract_memory")      # 关闭反思：答完直接收尾
```

三个关键点：
- **条件边 `add_conditional_edges`**：带一个路由函数 `route_after_reflect`，它读 state 返回暗号
  字符串，映射表把暗号翻译成下一个节点。即「看情况选路」的岔路口。
- **循环怎么形成**：`fuse → answer → reflect →(不足)→ augment_retrieve → fuse`，第 71 行
  `augment_retrieve → fuse` 把箭头**指回上游已走过的 fuse**，于是兜成环。这是 LangGraph
  区别于 LangChain 链的核心——能往回连、能画环。
- **防死循环**：`route_after_reflect` 内部判断 `revisions >= MAX_REVISIONS`，到上限强制走
  extract_memory，断开环。

## 7. 收尾 + 编译

```python
g.add_edge("extract_memory", END)    # 抽取记忆后到终点
return g.compile(checkpointer=checkpointer, store=store)
```

`compile(...)` 把「图纸」编译成可 `.invoke()` 的对象，并挂上两套记忆。返回的就是 `server.py`
里那个 `GRAPH`。两个出口（充分 / 次数用完）都汇到 `extract_memory → END`，保证一定结束。

## 这个文件浓缩的 LangGraph 知识点

| 看到的 | 概念 |
|--------|------|
| `StateGraph(MedState)` | 状态图 + 状态结构 |
| `add_node` | 注册节点 |
| `add_edge(A, B)` | 固定边（单行道）|
| `add_conditional_edges(节点, 路由函数, 映射)` | 条件边（岔路口）|
| `augment_retrieve → fuse` 接回前面 | 循环 |
| `compile(checkpointer=, store=)` | 编译 + 挂记忆 |
| `START` / `END` | 入口 / 出口哨兵 |

## 自测题（med_graph）

1. `add_edge` 和 `add_conditional_edges` 的区别？

2. 反思的循环是怎么形成的？哪一行是闭环关键？

3. `use_reflect=False` 时图会怎么走？

4. checkpointer/store 在函数里创建，为什么能被整个服务共享？

   关键不在"函数里创建"，而在**server.py 只调用一次 get_med_graph() 并把结果存成全局变量复用**：                                                                                                                                                                                                                   载时执行一次）                                                                                                                                                                                                                                                                                 

     GRAPH = get_med_graph()   # 只编译一次，checkpointer/store 跟着这个 GRAPH 走                                                                                                                                                                                                                                      @app.post("/chat")                                                                                                                                                                                                                                                                                         def chat(req):                                                                                                                                                                                                                                                                                                  GRAPH.invoke(...)     # 每个请求都复用同一个 GRAPH → 同一套记忆                                                                                                                                                                                                                                               所以整个服务进程里所有请求共用这一个 GRAPH，自然共用它身上那套 checkpointer/store。                                                                                                                                                                                                                               ▎ 反证：如果你在每个请求里都 get_med_graph() 一次，那每次都是全新的空记忆，就不共享了——记忆立刻失效。所以"编译一次、全局复用"是记忆能跨请求生效的前提。
