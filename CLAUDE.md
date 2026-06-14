# CLAUDE.md — 医疗知识助手

给 Claude Code 的项目上下文。开始任何任务前先读本文件，并遵循其中的约束。

## 项目目标

用 LangGraph 构建一个「医疗知识助手」：用户提一个（可能跨多方面的）医疗问题，Agent 规划
检索步骤、在内部医学教材库（Qdrant 混合检索）与外部 Web（Tavily 联网）两路找证据、融合
重排，最后由 DeepSeek 给出**带 `[编号]` 引用、严格防幻觉、可溯源**的回答。

定位：一个落地的检索增强（RAG）+ Agent 应用——混合检索 + 知识融合 + 带引用作答，
可 CLI / Web / Docker 运行。逐层演进：基线 → Planning → 知识融合 → 部署 →（后续）反思 / 记忆。

> ⚠️ 医疗高风险场景：作答只依据检索到的资料，资料不足必须如实说明，对外一律附免责声明
> 「仅供学习演示，非医疗建议」。

## 图主干（当前）

```
question → contextualize(用 history 指代消解 → standalone_question)   [短期记忆]
        → recall_memory(按 user_id 从 Store 召回健康事实)             [长期记忆·读]
        → plan → retrieve_internal(Qdrant 教材)
               → retrieve_external(Tavily Web，用户开关 use_external)
               → fuse(合并去重 → 统一重排 top-k)
               → answer(结合用户背景 + 证据，带 [编号] 引用，安全提示)
               → extract_memory(抽取本轮健康事实写回 Store)           [长期记忆·写] → END
```

记忆挂载（`med_graph.py`）：`compile(checkpointer=InMemorySaver(), store=InMemoryStore(index=BGE))`；
invoke 传 `config={"configurable": {"thread_id": session_id, "user_id": user_id}}`。内存版重启清空。

## 四大支柱 → LangGraph 映射

- **Planning**：`plan` 节点把复合问诊**平行拆解**成 1-3 个方面子问题（机理 / 用药 / 副作用…）。
  桥接型（后一跳依赖前一跳答案）的链式精化设计见 `docs/P2_5_DESIGN.md`。
- **Tool Use**：内部混合检索（`kb_search`：BGE 向量 + BM25 + RRF + 重排）+ 外部 Tavily（`external`）。
- **Memory（已实现 P5）**：短期 `InMemorySaver`（checkpointer，按 thread_id）撑会话内多轮，
  `contextualize` 节点做指代消解；长期 `InMemoryStore`（BGE 语义检索，按 user_id namespace），
  `recall_memory` 召回 + `extract_memory` 自动抽取用户过敏史/慢病/用药，作答时注入做安全提示。
  内存版重启清空（生产可换 SqliteSaver + Qdrant-backed Store）。
- **Reflection（路线图 P4）**：answer 后加 `reflect` 自判证据是否充分，不足则换源补检索，带修订上限。

## 技术栈与硬约束

**LLM：DeepSeek V4 云端 API，经 langchain-deepseek 的 ChatDeepSeek 接入。**
- 默认 `deepseek-v4-flash`（快、便宜，支持工具调用与结构化输出）。难节点可选 `deepseek-v4-pro`。
- 不要用 `deepseek-chat` / `deepseek-reasoner`——旧别名 2026-07-24 停用。
- DeepSeek 思考内容在独立的 reasoning_content 字段，`.content` 是干净的，不需要剥 `<think>`。
- 上下文缓存默认开启：稳定系统提示与重复检索上下文放 prompt 前部并保持一致，可命中缓存降本。

**向量化 + 重排：Xinference 上的 BGE（本地 GPU，经 SSH 隧道 9997 或 host.docker.internal）。**
- DeepSeek 没有 embedding 接口，这部分留在 Xinference，别用 DeepSeek 做向量化。
- `XinferenceEmbeddings` 用 model_uid（不是模型名）。
- 重排没有现成 LangChain 封装，打 `/v1/rerank` REST（见 `embeddings.py`），不要自己造别的。

**向量库 + 混合检索：Qdrant。**
- 命名向量集合：`dense`（bge-m3，1024 维 COSINE）+ `sparse`（FastEmbed BM25，IDF 由 Qdrant 库侧施加）。
- 检索：dense + sparse 双路 Prefetch → Qdrant 原生 RRF 融合 → BGE 重排取 top-k（见 `kb_search.py`）。
- 双模式：开发期本地落盘（`QDRANT_PATH`）；部署期连服务器（设 `QDRANT_URL`）。只切环境变量，零代码改动。

**外部源：Tavily Web 搜索（`external.py`）。** 联网开关 `use_external`；失败 try/except 优雅降级仅用内部源。

**编排：LangGraph。** StateGraph 组装节点；后续反思回路用条件边，记忆用 checkpointer + Store。

**服务 + 部署：FastAPI（`server.py`）+ 网页前端（`static/`）+ Docker（`docker-compose.yml` 编排 app + qdrant）。**
- Xinference 留宿主机（GPU），app 容器经 `host.docker.internal:9997` 连。详见 `docs/P6_DEPLOY.md`。

## 目录

```
config.py        端点 / 模型 / UID（.env 覆盖，唯一配置入口）
_bootstrap.py    放行 OpenMP 重复加载（入口最先 import）
llm.py           ChatDeepSeek 封装
embeddings.py    Xinference BGE 向量化 + 重排
sparse.py        FastEmbed BM25 稀疏向量
vectordb.py      Qdrant 客户端工厂（本地/服务器双模式）
ingest.py        灌库管道（切块 → dense+sparse → 写 Qdrant）
kb_search.py     内部混合检索（dense+sparse RRF + 重排）
external.py      Tavily Web 搜索
med_state.py     图状态 MedState
med_nodes.py     节点：plan / retrieve_internal / retrieve_external / fuse / answer
med_graph.py     组装 StateGraph + compile
med_rag.py       CLI 入口
server.py        FastAPI 服务（/ /health /chat）
static/index.html 网页前端
compare_retrieval.py / dedup_kb.py   检索对比 / 去重工具
Dockerfile / docker-compose.yml / .dockerignore   容器化
```

## 评测

医疗答复是开放长文本，EM/F1 失效。用**检索 recall@k**（金标 chunk 是否召回）+ **LLM-as-judge**
评答案质量。见 `docs/P1_DESIGN.md`、`docs/HYBRID_RETRIEVAL_EVIDENCE.md`。

## 约定

- 写代码前先读相关已有文件；所有端点 / 模型 / 密钥只从 `config.py` 取，不在别处硬编码。
- 密钥只放 `.env`，绝不写进代码、日志或提交。
- 解析模型输出优先用结构化输出（DeepSeek 支持 JSON / 工具调用），不要用脆弱的正则去抠。
- 医疗作答：只依据资料、不编造、不足明说、强制 `[编号]` 引用、附免责声明。
- 入口脚本第一行 `import _bootstrap`（放行 OpenMP）再 import torch/onnx 相关。
- 改完跑一遍相关脚本验证（`med_rag.py` / `kb_search.py` / `server` 健康检查）再说「完成」。
- 每完成一个小阶段，提醒用户 git commit。
