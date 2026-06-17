# 医疗 RAG（落地应用）项目解读 + 面试速查

内外部知识融合的中文医疗问答系统的完整解读。配合 `docs/DOCKER_INTERVIEW.md`
（部署）、`docs/P6_DEPLOY.md`（部署步骤）阅读。

## 一、一句话定位

带引用、内外部知识融合的中文医疗问答系统：用户问一句 → 规划拆解 → 在「内部医学
教材库（Qdrant 混合检索）」+「外部 Web（Tavily）」两路找证据 → 融合重排 →
DeepSeek 带 `[编号]` 引用作答，全程 LangGraph 编排，FastAPI + 网页前端 + Docker 部署。

## 二、整体数据流

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

图结构（`med_graph.py`）：`START → plan → retrieve_internal → retrieve_external
→ fuse → answer → END`（线性图，外部检索靠节点内 `if use_external` 短路）。

## 三、各文件职责（按分层理解）

### 第 0 层：配置与底座

| 文件 | 做什么 | 关键点 |
|------|--------|--------|
| `config.py` | 唯一配置入口，端点/模型/UID/超参，`.env` 可覆盖 | `QDRANT_URL` 设了连服务器、不设走本地落盘——本地/部署双模式开关 |
| `_bootstrap.py` | 设 `KMP_DUPLICATE_LIB_OK=TRUE` | 必须在 import torch/onnx 之前，否则 Windows OpenMP 重复加载崩溃(OMP #15) |
| `llm.py` | `ChatDeepSeek` 封装 | DeepSeek 思考走 `reasoning_content`，`.content` 干净，不用剥 `<think>` |

### 第 1 层：检索基础设施（Tool Use 支柱）

| 文件 | 做什么 | 关键点 |
|------|--------|--------|
| `embeddings.py` | Xinference BGE 向量化 + 重排 | 向量化用 `XinferenceEmbeddings`(传 model_uid)；重排打 `/v1/rerank` REST |
| `sparse.py` | BM25 稀疏向量（混合检索的词面路） | FastEmbed 本地 BM25；文档侧长度归一化，查询侧不做，IDF 交 Qdrant 库侧 |
| `vectordb.py` | Qdrant 客户端工厂 | 单独成模块，避免检索路径间接 import text_splitters 与 torch 加载冲突段错误 |
| `ingest.py` | 灌库管道：读教材→切块→算 dense+sparse→写 Qdrant | 一个 chunk 存两向量；`--recreate` 本地模式直接删目录彻底清库 |
| `kb_search.py` | 内部检索核心：混合召回 + 重排 | dense+sparse 两路 Prefetch → Qdrant 原生 RRF 融合 → rerank 取 top-k |

### 第 2 层：外部知识源（知识融合）

| 文件 | 做什么 |
|------|--------|
| `external.py` | Tavily Web 搜索封装，返回 `{text,title,url}`，补教材外时效信息 |

### 第 3 层：Agent 编排（Planning + 融合）

| 文件 | 做什么 |
|------|--------|
| `med_state.py` | 图状态 `MedState`(TypedDict)：question/use_external/子问题/内外部证据/融合证据/answer |
| `med_nodes.py` | 5 个节点：plan / retrieve_internal / retrieve_external / fuse / answer |
| `med_graph.py` | `StateGraph` 把 5 节点连成图并 `compile()` |

### 第 4 层：对外服务

| 文件 | 做什么 |
|------|--------|
| `med_rag.py` | CLI 入口：`python med_rag.py "问题" --web` |
| `server.py` | FastAPI：`GET /`(前端)、`GET /health`、`POST /chat`；启动时编译图一次复用 |
| `static/index.html` | 网页前端（带联网开关） |
| `Dockerfile`/`docker-compose.yml`/`.dockerignore` | 容器化部署 |

### 辅助工具

| 文件 | 做什么 |
|------|--------|
| `compare_retrieval.py` | 混合检索 vs 纯 dense 对比实证 |
| `dedup_kb.py` | 知识库去重 |

## 四、最能体现工程功力的设计点

1. **混合检索二阶段架构**：召回(dense 语义 + sparse BM25 词面，RRF 融合，广撒网
   RECALL_K=20) → 精排(BGE reranker cross-encoder 取 top-3)。dense 抓意思相近、
   sparse 抓药名/专有名词精确匹配，互补。工业级 RAG 标准做法。
2. **融合后统一重排**：`fuse` 把内外部证据混在一起对原问题重排，最相关的浮上来，
   不论来源，而非机械各取几条。
3. **优雅降级**：外部检索 try/except，Tavily 挂了返回空、仅用内部源，服务不崩。
4. **配置与代码解耦**：`QDRANT_URL` 一个环境变量切本地/服务器，零代码改动。
5. **结构化输出而非正则**：plan 用 `with_structured_output(method="json_mode")`
   让 DeepSeek 直接吐合法 JSON。
6. **防幻觉 prompt**：系统提示明确只依据资料、不编造、不足说不足、冲突指出，强制
   `[编号]` 引用——医疗场景至关重要。

## 五、面试问答

### A. 项目整体 / 业务理解
1. 解决什么问题？→ 通用 LLM 答医疗会幻觉、无依据；RAG 把答案锚定可信教材+实时 Web，强制引用可溯源。
2. 为什么医疗特别需要 RAG + 引用？→ 容错率低，幻觉有害；引用可核查，资料不足如实说明。
3. 内部教材 vs 外部 Web 作用？→ 教材是稳定基础知识，Web 补时效/最新指南/教材外内容，冲突时提示分歧。

### B. 检索技术（核心考点）
4. 为什么混合检索不直接向量检索？→ dense 抓语义、sparse(BM25) 抓精确词面，互补；纯 dense 漏精确匹配。
5. RRF 怎么融合两路？→ Reciprocal Rank Fusion，按各路排名倒数加权，无需归一化不同打分尺度，Qdrant 原生支持。
6. 召回与重排为什么分两步？→ 召回用快的双塔向量广撒网，重排用慢但准的 cross-encoder 精排，兼顾速度精度。
7. 为什么 embedding 用 Xinference 不用 DeepSeek？→ DeepSeek 没 embedding 接口，BGE 留 Xinference(GPU)。
8. chunk 大小怎么定？→ BGE max_tokens=512，中文约 1 字≈1 token，控制 400 字内最稳。

### C. Agent / LangGraph
9. 为什么用 LangGraph 不简单串函数？→ 节点化+状态管理，便于加条件边/可视化/接 LangSmith，为 Reflection/Memory 留扩展点。
10. planner 拆子问题有什么用？→ 复合问题拆成多方面分别检索，覆盖更全。
11. 怎么保证不编造？→ 结构化输出 + 严格 prompt。

### D. 工程与部署
12. 本地与线上怎么切换？→ 环境变量 `QDRANT_URL`，12-Factor。
13. 怎么部署？→ Docker compose 编排 app+qdrant，Xinference 留宿主机(GPU)，密钥 .env 运行时注入。
14. 外部 API 挂了怎么办？→ try/except 优雅降级仅用内部源。
15. `KMP_DUPLICATE_LIB_OK` 坑？→ torch 和 onnxruntime 都带 OpenMP 重复加载崩溃，`_bootstrap.py` 最先导入放行。

### E. 短板 / 改进（提前准备，主动说显诚实）
16. 检索质量怎么评估？→ 医疗这条线目前无 HotpotQA 那样的客观 EM/F1，可补医疗 QA 测试集 + 召回率/引用准确率指标。
17. 怎么防过时/错误医疗信息？→ 来源标注 + 冲突提示 + 免责声明；可加来源可信度加权。
18. 并发/性能？→ 启动编译图一次复用；可加缓存(DeepSeek 上下文缓存)、qdrant 服务器版 HNSW 索引。

## 六、一句话总结（面试开场）

> 我做了一个内外部知识融合的医疗 RAG 系统：内部用 Qdrant 做 dense+sparse 混合检索
> 加 BGE 重排，外部用 Tavily 联网补时效信息，两路证据融合后统一重排，再让 DeepSeek
> 带引用作答、严格防幻觉。整体用 LangGraph 编排成 plan→检索→融合→作答的流程，对外
> 是 FastAPI + 网页前端，用 Docker compose 部署，靠环境变量实现本地开发到容器部署零改动切换。