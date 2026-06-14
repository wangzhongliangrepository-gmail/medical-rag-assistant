# 项目面试速查（合订本）

医疗 RAG（落地应用）这条线的面试准备。三部分：
- **第一部分 Agent 概念**：四大支柱、主流范式、LangGraph
- **第二部分 医疗 RAG 项目解读**：结构、各文件、检索技术
- **第三部分 Docker 部署**：容器架构、Dockerfile/compose、上云

配合 `docs/P6_DEPLOY.md`（部署步骤）阅读。

---

# 第一部分：Agent 概念

## 1.1 什么是 Agent

普通 LLM 调用是「prompt → 文本」一锤子买卖；Agent 让 LLM 成为**能自主决策、调
工具、多步骤完成任务**的系统，LLM 从「答题者」变成「决策中枢」。

> 一句话：**Agent = LLM（大脑）+ 工具（手脚）+ 记忆（上下文）+ 循环（反思纠错）**。

## 1.2 Agent 四大支柱（对应本医疗助手）

| 支柱 | 是什么 | 本项目状态 |
|------|--------|-----------|
| **Planning** | 把复杂问题拆成子步骤 | ✅ `med_nodes.py` 的 `plan` 节点把复合问诊拆成 1-3 个方面（机理/用药/副作用…）分别检索 |
| **Tool Use** | 调外部工具获取自身没有的信息 | ✅ 混合检索(Qdrant dense+sparse+RRF+rerank) + Tavily 联网搜索，两路证据 `fuse` 融合 |
| **Memory** | 跨轮/跨会话保留上下文与经验 | ✅ 短期 `InMemorySaver`（会话内多轮，`contextualize` 指代消解）+ 长期 `InMemoryStore`（BGE 语义检索，按 user_id 记过敏史/慢病，`recall_memory` 召回 + `extract_memory` 自动抽取，作答做安全提示） |
| **Reflection** | 自我批判、判断证据够不够、不够重来 | 🚧 路线图 P4（Reflexion 回路：答完自判证据充分性，不足换源补检索，带修订上限） |

> 当前医疗助手已落地 **Planning + Tool Use + 知识融合 + Memory**；Reflection 是规划中的
> 下一个支柱（见 README 路线图）。

## 1.3 主流 Agent 范式

- **ReAct**：Thought→Action→Observation 循环。灵活但慢、贵、易跑偏。
- **Plan-and-Execute**（本项目 Planning 用的）：先一次性规划再执行，可控、调用少。
  本项目是**平行拆解**——把问诊拆成互相独立的方面同时检索。
- **Reflexion**（路线图 P4）：执行后自我反思→发现证据不足→换源/补检索重试，带修订次数上限防死循环。

## 1.4 LangGraph 概念

| 概念 | 是什么 | 本项目对应 |
|------|--------|-----------|
| State | 节点间流动的共享数据 | `MedState` TypedDict |
| Node | 读 state 改 state 的函数 | contextualize / recall_memory / plan / retrieve_* / fuse / answer / extract_memory |
| Edge | 节点连接 | `add_edge("plan","retrieve_internal")` |
| Conditional Edge | 按 state 决定走向 | 路线图：reflect 判断够→结束/不够→补检索 |
| Checkpointer | 持久化 state(短期记忆) | ✅ `InMemorySaver`，按 thread_id 撑会话内多轮（history reducer 累积） |
| Store | 跨线程长期记忆 | ✅ `InMemoryStore`（BGE 语义检索），按 user_id 记用户健康事实 |

> 为什么用 LangGraph 不用 if/else 串函数？→ 把 Agent 建模成图，条件边天然支持
> 「反思回路」这种循环，自带状态管理、可视化、LangSmith 追踪，为后续加 Reflection/Memory 留扩展点。

## 1.5 Agent 面试问答

**概念**
1. Agent vs 普通 RAG？→ 普通 RAG 是「检索一次→作答」固定流程；Agent 能先规划多步、调多个工具、（加上反思后）自判够不够再重试。
2. Agent vs Workflow？→ Workflow 是人预编排固定路径；Agent 让 LLM 自主决策路径；实践常混合（本项目偏「LLM 决策 + 图约束」）。
3. ReAct vs Plan-and-Execute？→ ReAct 边想边做灵活但慢；P&E 先规划后执行可控但不灵活。本项目用 P&E 做平行拆解。

**工具调用**
4. Function Calling 怎么工作？→ LLM 输出结构化工具调用意图(JSON)，框架执行，结果回喂 LLM。本项目 plan 用 `with_structured_output(method="json_mode")`。
5. 工具失败怎么办？→ Tavily try/except 优雅降级，仅用内部源不崩。
6. 怎么防乱调工具/幻觉参数？→ 结构化输出约束 schema + 严格 prompt。

**规划**
7. 怎么拆解复杂问题？→ plan 节点把复合问诊拆成 1-3 个方面；单一问题就只输出原问题。
8. 拆解一定更好吗？→ 不一定！平行拆解适合「多方面」问题；桥接型（后一跳依赖前一跳答案）需链式精化，否则悬空子问题检索全是噪声（见 `P2_5_DESIGN.md`）。

**记忆（已实现 P5）**
9. 短期 vs 长期记忆怎么分？→ 短期=会话内多轮，`InMemorySaver` checkpointer 按 thread_id 持久化 state（含 history）；长期=跨会话，`InMemoryStore` 按 user_id namespace 存用户健康事实。
10. 多轮指代消解怎么做？→ `contextualize` 节点用最近 N 轮 history 把「那它的禁忌呢」改写成自包含的「二甲双胍的禁忌」，下游检索全用改写后的问题。
11. 长期记忆怎么写、怎么读？→ 写：`extract_memory` 用结构化输出自动抽取用户明确陈述的过敏/慢病/用药，`store.put`；读：`recall_memory` 用 `store.search(query=)` BGE 语义召回，注入作答 prompt 做安全提示。
12. 记忆怎么不误抽 / 不串用户？→ 抽取 prompt 严格限「只记明确陈述、没有返回空」；user_id 做 namespace 隔离；抽取失败 try/except 不影响作答。

**反思（路线图 P4）**
13. Reflection 打算怎么实现？→ answer 后加 reflect 节点自判证据充分性，不足则换源补检索，条件边回检索，带修订上限防死循环。

**评估**
11. 医疗答案怎么评估？→ 开放长文本 EM 失效，用**检索 recall@k**（金标 chunk 是否召回）+ **LLM-as-judge** 评答案质量（见 `P1_DESIGN.md`）。
12. 怎么防幻觉？→ 结构化输出 + 严格 prompt（只依据资料、不编造、不足说不足、冲突指出）+ 强制 `[编号]` 引用可溯源。

---

# 第二部分：医疗 RAG 项目解读

## 2.1 一句话定位

带引用、内外部知识融合的中文医疗问答系统：用户问一句 → 规划拆解 → 在「内部医学
教材库(Qdrant 混合检索)」+「外部 Web(Tavily)」两路找证据 → 融合重排 → DeepSeek
带 `[编号]` 引用作答，LangGraph 编排，FastAPI + 网页前端 + Docker 部署。

## 2.2 整体数据流

```
用户问题
   │
   ▼
[plan]  DeepSeek 把问题拆成 1-3 个子问题（结构化 JSON 输出）
   │
   ├──────────────────────┬─────────────────────────┐
   ▼                      ▼
[retrieve_internal]   [retrieve_external]
 对每个子问题：         若用户开了联网开关：
 Qdrant 混合检索       Tavily Web 搜索
 (dense+sparse        (失败优雅降级为空)
  →RRF→重排)
   │                      │
   └──────────┬───────────┘
              ▼
          [fuse]  内外部证据合并 → 对原问题统一重排 → top-k
              ▼
          [answer]  DeepSeek 只依据资料作答，关键结论标 [编号]
              ▼
          带引用的答案 + 证据来源列表
```

图结构(`med_graph.py`)：`START → plan → retrieve_internal → retrieve_external →
fuse → answer → END`（线性图，外部检索靠节点内 `if use_external` 短路）。

## 2.3 各文件职责（按分层）

**第 0 层 配置与底座**

| 文件 | 做什么 | 关键点 |
|------|--------|--------|
| `config.py` | 唯一配置入口 | `QDRANT_URL` 设了连服务器、不设走本地落盘——本地/部署双模式开关 |
| `_bootstrap.py` | 设 `KMP_DUPLICATE_LIB_OK=TRUE` | 必须在 import torch/onnx 之前，否则 Windows OpenMP 崩溃(OMP #15) |
| `llm.py` | `ChatDeepSeek` 封装 | 思考走 `reasoning_content`，`.content` 干净不用剥 `<think>` |

**第 1 层 检索基础设施（Tool Use）**

| 文件 | 做什么 | 关键点 |
|------|--------|--------|
| `embeddings.py` | Xinference BGE 向量化 + 重排 | 向量化用 `XinferenceEmbeddings`(model_uid)；重排打 `/v1/rerank` REST |
| `sparse.py` | BM25 稀疏向量(词面路) | FastEmbed 本地 BM25；IDF 交 Qdrant 库侧 |
| `vectordb.py` | Qdrant 客户端工厂 | 单独成模块，避免间接 import text_splitters 与 torch 加载冲突段错误 |
| `ingest.py` | 灌库：读教材→切块→算 dense+sparse→写 Qdrant | 一 chunk 存两向量；`--recreate` 本地模式删目录彻底清库 |
| `kb_search.py` | 内部检索核心：混合召回+重排 | dense+sparse 双路 Prefetch → Qdrant RRF 融合 → rerank top-k |

**第 2 层 外部知识源**

| 文件 | 做什么 |
|------|--------|
| `external.py` | Tavily Web 搜索，返回 `{text,title,url}`，补教材外时效信息 |

**第 3 层 Agent 编排**

| 文件 | 做什么 |
|------|--------|
| `med_state.py` | 图状态 `MedState`(question/use_external/子问题/内外部证据/融合证据/answer) |
| `med_nodes.py` | 5 节点：plan/retrieve_internal/retrieve_external/fuse/answer |
| `med_graph.py` | `StateGraph` 连图并 `compile()` |

**第 4 层 对外服务**

| 文件 | 做什么 |
|------|--------|
| `med_rag.py` | CLI：`python med_rag.py "问题" --web` |
| `server.py` | FastAPI：`/`、`/health`、`/chat`；启动编译图一次复用 |
| `static/index.html` | 网页前端(带联网开关) |
| `Dockerfile`/`docker-compose.yml`/`.dockerignore` | 容器化 |

**辅助**：`compare_retrieval.py`(检索对比)、`dedup_kb.py`(去重)

## 2.4 工程亮点

1. **混合检索二阶段**：召回(dense 语义+sparse BM25 词面，RRF 融合，RECALL_K=20)
   → 精排(BGE reranker cross-encoder top-3)。dense 抓意思、sparse 抓药名精确匹配，互补。
2. **融合后统一重排**：内外部证据混在一起对原问题重排，最相关的浮上来不论来源。
3. **优雅降级**：Tavily try/except，挂了仅用内部源不崩。
4. **配置与代码解耦**：`QDRANT_URL` 一个变量切本地/服务器，零代码改动。
5. **结构化输出而非正则**：plan 用 `with_structured_output(method="json_mode")`。
6. **防幻觉 prompt**：只依据资料、不编造、不足说不足、冲突指出，强制 `[编号]` 引用。

## 2.5 医疗 RAG 面试问答

**业务理解**
1. 解决什么问题？→ 通用 LLM 答医疗会幻觉无依据；RAG 锚定可信教材+实时 Web，强制引用可溯源。
2. 为何医疗特别需要引用？→ 容错率低幻觉有害；可核查，资料不足如实说明。
3. 内部教材 vs 外部 Web？→ 教材稳定基础知识，Web 补时效/最新指南/教材外内容，冲突提示分歧。

**检索技术（核心）**
4. 为何混合检索不直接向量？→ dense 抓语义、sparse(BM25) 抓精确词面，互补；纯 dense 漏精确匹配。
5. RRF 怎么融合？→ Reciprocal Rank Fusion，按排名倒数加权，无需归一化不同打分尺度，Qdrant 原生。
6. 召回与重排为何分两步？→ 召回快的双塔向量广撒网，重排慢但准的 cross-encoder 精排，兼顾速度精度。
7. 为何 embedding 用 Xinference 不用 DeepSeek？→ DeepSeek 没 embedding 接口，BGE 留 Xinference(GPU)。
8. chunk 大小怎么定？→ BGE max_tokens=512，中文约 1 字≈1 token，400 字内最稳。

**工程**
9. KMP_DUPLICATE_LIB_OK 坑？→ torch 和 onnxruntime 都带 OpenMP 重复加载崩溃，`_bootstrap.py` 最先导入放行。

**短板/改进（主动说显诚实）**
10. 检索质量怎么评估？→ 医疗线目前无 HotpotQA 那样客观 EM/F1，可补医疗 QA 测试集+召回率/引用准确率。
11. 怎么防过时/错误医疗信息？→ 来源标注+冲突提示+免责声明；可加来源可信度加权。
12. 并发/性能？→ 启动编译图一次复用；可加缓存、qdrant 服务器版 HNSW 索引。

---

# 第三部分：Docker 部署

## 3.1 容器架构

```
┌─ docker-compose（一个内部网络）──────────┐
│  app   (FastAPI, :8000)   ← Dockerfile 构建  │
│  qdrant(向量库服务器, :6333) ← 官方镜像        │
└──────────┬───────────────────────────────┘
           │ host.docker.internal:9997
   ┌───────┴─────────┐
   │ Xinference (GPU) │  留宿主机（BGE 嵌入+重排）
   └─────────────────┘
   DeepSeek / Tavily   云端 API（密钥经 .env 注入）
```

| 层 | 放哪 | 为什么 |
|----|------|--------|
| app + qdrant | 容器 | qdrant 服务器版有真正 HNSW 索引，解决本地落盘 >2万点性能告警 |
| Xinference(BGE) | 宿主机 | 要 GPU，Windows GPU 透传成本高；app 经 `host.docker.internal` 连 |
| DeepSeek/Tavily | 云端 API | 本就是外部服务，密钥运行时注入不烤进镜像 |

核心思想：**配置与代码解耦(12-Factor)**——同一份代码靠环境变量在本地/容器间零改动切换。

## 3.2 Dockerfile 设计点

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get install gcc g++       # fastembed/onnxruntime 编译依赖
COPY requirements.txt .           # ① 先拷依赖清单
RUN pip install -r requirements.txt   # ② 装包（最慢一层）
COPY . .                          # ③ 再拷代码
ENV KMP_DUPLICATE_LIB_OK=TRUE
EXPOSE 8000
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000"]
```

- **分层缓存**：①②③ 分开，requirements 不变就复用 pip 缓存，改代码不重装包。
- **`--host 0.0.0.0`**：容器内 `127.0.0.1` 宿主机访问不到，必须监听 0.0.0.0。

## 3.3 compose 关键点

- **服务名当主机名**：`QDRANT_URL=http://qdrant:6333`，compose 内部网络 DNS。
- **`host.docker.internal`**：容器访问宿主机(Xinference)；Linux 需 `extra_hosts: host-gateway`。
- **Volume**：容器无状态，Volume 把 qdrant 数据存容器外，重建仍在。
- **`depends_on`**：只保证启动顺序，不等服务 ready，严格需 healthcheck。

## 3.4 部署步骤

前提：宿主机 Xinference 已加载 `bge-m3`+`bge-reranker-v2-m3`，`.env` 密钥齐全。

```powershell
docker compose up -d qdrant                  # 1. 起向量库服务器
$env:QDRANT_URL="http://localhost:6333"       # 2. 灌库(写进容器 qdrant，走宿主机 Xinference)
python ingest.py --recreate
docker compose up -d --build app             # 3. 构建并起 app
# 4. 访问 http://localhost:8000 (前端) / /docs (Swagger) / /health
```

> 换 Docker 后必须重新 ingest：本地落盘 `./qdrant_db` 与容器 qdrant 是两套独立存储，且 `.dockerignore` 已排除 `qdrant_db/`。

## 3.5 上云

- app + qdrant 镜像可 push 到仓库(阿里云 ACR)，云主机 pull 运行。
- **GPU 卡点**：Xinference 要 GPU，普通云主机没有。三选一：①租 GPU 实例；
  ②嵌入/重排换云端 API(改 embeddings.py 地址)，最省钱；③qdrant 用 Qdrant Cloud + 嵌入用云端 API，全程无 GPU。
- 上云要改：`XINFERENCE_URL`/`QDRANT_URL` 指向真实地址；云主机单独配 `.env`；云上 qdrant 是空的需重新灌库。

## 3.6 Docker 面试问答

1. 镜像 vs 容器？→ 镜像是模板(类)，容器是运行实例(对象)。
2. 为何先拷 requirements 再拷代码？→ 分层缓存，避免改代码重装包。
3. EXPOSE vs ports？→ EXPOSE 仅文档声明，对外靠 compose `ports: 8000:8000`。
4. 镜像怎么瘦身？→ slim 基础镜像、`--no-cache-dir`、清 apt 缓存；进阶多阶段构建。
5. 为何 Xinference 不进容器？→ GPU 透传成本，权衡取舍。
6. 容器怎么连宿主机服务？→ `host.docker.internal`。
7. 一份代码怎么同时支持本地+容器？→ 环境变量 + `os.getenv`，12-Factor。
8. 密钥怎么管不进镜像？→ `.dockerignore` 排除 + 运行时 `env_file` 注入。
9. 容器挂了数据丢吗？→ Volume 持久化。
10. depends_on 保证依赖就绪吗？→ 不能，要 healthcheck。
11. 怎么上云？→ 镜像推仓库云主机 compose up；GPU 卡点见 3.5。

---

# 附：两条一句话总结（面试开场可用）

**Agent 能力**：
> 我用 LangGraph 在医疗助手上落地了 Agent 支柱：Planning 用 plan-and-execute 把复合
> 问诊平行拆成多方面，Tool Use 是 Qdrant 混合检索 + Tavily 联网两路工具、再 fuse 融合，
> Memory 用 checkpointer 撑会话内多轮指代消解、用带 BGE 语义检索的 Store 跨会话记住用户
> 过敏史/慢病并在作答时做安全提示。作答严格防幻觉、带引用可溯源；Reflection（证据自检
> 补检索）是规划中的下一支柱。

**医疗 RAG 项目**：
> 内外部知识融合的医疗 RAG：内部 Qdrant 做 dense+sparse 混合检索加 BGE 重排，外部
> Tavily 联网补时效，两路证据融合后统一重排，DeepSeek 带引用作答严格防幻觉。
> LangGraph 编排 plan→检索→融合→作答，FastAPI+网页前端，Docker compose 部署，
> 靠环境变量实现本地到容器零改动切换。