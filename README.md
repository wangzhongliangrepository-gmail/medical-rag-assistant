# 医疗知识助手（医疗 RAG Agent）

内外部知识融合的中文医疗问答助手：用户提一个（可能跨多方面的）医疗问题，Agent 规划检索、
在**内部医学教材库**（Qdrant 混合检索）和**外部 Web**（Tavily 联网）两路找证据，融合重排后由
**DeepSeek 带 `[编号]` 引用作答**，严格防幻觉、可溯源。用 LangGraph 编排，对外是 FastAPI +
网页前端，用 Docker compose 部署。

> ⚠️ 本项目仅供学习演示，**非医疗建议**；如有健康问题请咨询专业医师。

## 整体数据流

```
用户问题
   │
   ▼
[plan]  把复合问诊拆成 1-3 个方面子问题（结构化 JSON 输出）
   │
   ├──────────────────────┬─────────────────────────┐
   ▼                      ▼
[retrieve_internal]   [retrieve_external]
 对每个子问题：         （用户开启联网开关时）
 Qdrant 混合检索       Tavily Web 搜索
 dense+sparse→RRF→重排  失败优雅降级为空
   │                      │
   └──────────┬───────────┘
              ▼
          [fuse]  内外部证据合并 → 对原问题统一重排 → top-k
              ▼
          [answer]  只依据资料作答，关键结论标 [编号]，不足如实说明
              ▼
          带引用的答案 + 证据来源列表
```

## 技术栈

- **LLM**：DeepSeek V4 云端 API（`langchain-deepseek` 的 `ChatDeepSeek`），默认 `deepseek-v4-flash`。
- **向量化 + 重排**：Xinference 上的 BGE（`bge-m3` 向量 + `bge-reranker-v2-m3` 重排，本地 GPU）。
- **向量库 + 混合检索**：Qdrant（dense 语义 + sparse BM25 → RRF 融合 → BGE 重排）。
- **外部源**：Tavily Web 搜索（联网补时效/最新指南）。
- **编排**：LangGraph（StateGraph）。
- **服务 + 部署**：FastAPI + 原生网页前端 + Docker（compose 编排 app + qdrant）。

## 目录结构

```
config.py        端点 / 模型 / UID（.env 覆盖，唯一配置入口）
_bootstrap.py    放行 OpenMP 重复加载（入口最先 import）
llm.py           ChatDeepSeek 封装
embeddings.py    Xinference BGE 向量化 + 重排（/v1/rerank REST）
sparse.py        FastEmbed BM25 稀疏向量（混合检索的词面路）
vectordb.py      Qdrant 客户端工厂（本地落盘 / 服务器双模式）
ingest.py        灌库管道：教材 → 切块 → dense+sparse → 写 Qdrant
kb_search.py     内部检索：混合召回（dense+sparse RRF）+ BGE 重排
external.py      Tavily Web 搜索
med_state.py     图状态 MedState
med_nodes.py     节点：plan / retrieve_internal / retrieve_external / fuse / answer
med_graph.py     组装 StateGraph 并 compile
med_rag.py       CLI 入口：python med_rag.py "问题" [--web]
server.py        FastAPI 服务：/ /health /chat
static/index.html 网页前端（带联网开关）
compare_retrieval.py  混合检索 vs 纯 dense 对比工具
dedup_kb.py      知识库去重工具
Dockerfile / docker-compose.yml / .dockerignore   容器化部署
fastembed_cache/ 内置 BM25 模型缓存（容器离线构建用）
docs/            设计与部署文档
```

## 快速开始

### 前置
1. `platform.deepseek.com` 申请 API 密钥，填进 `.env` 的 `DEEPSEEK_API_KEY`。
2. Xinference 启动 `bge-m3`（向量）与 `bge-reranker-v2-m3`（重排），记下 UID 填进 `.env`。
3. （可选联网）`tavily.com` 申请密钥填 `TAVILY_API_KEY`。

### 本地运行
```bash
pip install -r requirements.txt
cp .env.example .env          # 填密钥与模型 UID

python ingest.py --recreate   # 灌库（首次，走 Xinference 向量化）
python med_rag.py "二甲双胍的副作用和禁忌"          # CLI 问答
python med_rag.py "高血压一线药的副作用" --web      # 融合联网

uvicorn server:app --reload   # 起 Web 服务 → http://localhost:8000
```

### Docker 部署
```bash
docker compose up -d qdrant            # 1. 起向量库服务器
export QDRANT_URL=http://localhost:6333 && python ingest.py --recreate  # 2. 灌库
docker compose up -d --build app       # 3. 起 app → http://localhost:8000
```
详见 `docs/P6_DEPLOY.md`。**本地落盘 → 容器部署只切环境变量 `QDRANT_URL`，零代码改动。**

## 进度路线（P0 → P6）

| 阶段 | 内容 | 状态 |
|------|------|------|
| **P0** | Qdrant 持久向量库 + 医疗 KB 灌库 + 混合检索（dense+sparse+RRF+rerank） | ✅ |
| **P1** | 医疗 RAG 基线（问诊 → 混合检索 → 带引用作答） | ✅ |
| **P2** | +Planning（复合问题拆方面 → 分方面检索 → 汇总） | ✅ |
| **P3** | 知识融合（内部教材 KB + 外部 Tavily Web，带联网开关） | ✅ |
| **P6** | FastAPI 服务 + 网页前端 + Docker 容器化（已端到端验证） | ✅ |
| **P4** | +Reflection（答完自判证据充分性，不足换源补检索） | 🚧 待做 |
| **P5** | +Memory（短期 checkpointer 多轮 + 长期 Store 用户记忆/事实缓存） | 🚧 待做 |

## 评测

医疗答复是开放长文本，EM/F1 基本失效。采用：**检索 recall@k**（金标 chunk 是否被召回）+
**LLM-as-judge** 评答案质量。详见 `docs/P1_DESIGN.md`、`docs/HYBRID_RETRIEVAL_EVIDENCE.md`。

## 文档

- `docs/INTERVIEW.md` — 面试速查（Agent 概念 + 项目解读 + Docker 部署）
- `docs/P0_WALKTHROUGH.md` — 灌库与 Qdrant 操作走查
- `docs/P1_DESIGN.md` / `docs/P2_5_DESIGN.md` — 基线与多跳设计
- `docs/HYBRID_RETRIEVAL_EVIDENCE.md` — 混合检索实证
- `docs/P6_DEPLOY.md` — Docker 部署详解
