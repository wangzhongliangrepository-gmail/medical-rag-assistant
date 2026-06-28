# P0 设计：检索底座（灌库 + 混合检索）

## 目标

给医疗 RAG 建一个**可溯源、召回稳**的检索底座：把医学教材语料灌进向量库，
支持一条查询召回最相关的若干段教材，并带回**来源元数据**（供后续带 `[编号]` 引用）。

## 为什么是「混合检索」而非纯向量

医疗语料里**专有名词密集**（药名、疾病名、指标缩写如 GFR/HbA1c）。两路互补：

| 路 | 模型 | 擅长 | 短板 |
|----|------|------|------|
| **dense** | bge-m3（1024 维，COSINE） | 语义近义、换种说法也能召回 | 专有名词/缩写易被语义"抹平"漏召 |
| **sparse** | FastEmbed BM25（词面） | 精确词面匹配，药名/术语命中稳 | 不懂同义改写 |

纯 dense 会漏掉「就要这个词」的精确匹配，纯 BM25 不懂换说法 → **两路并召再融合**最稳。

## 实现（`ingest.py` / `kb_search.py`）

**灌库**：`medical_book_zh.json`（JSONL）→ `RecursiveCharacterTextSplitter`
（`chunk_size=400` / `overlap=80`，中文分隔符 `。！？；，`）→ 每个 chunk 同时算 dense + sparse →
写进 Qdrant **命名向量集合**：

```
collection: medical_kb
  dense  : VectorParams(size=1024, distance=COSINE)
  sparse : SparseVectorParams(modifier=IDF)   # BM25 的 IDF 由 Qdrant 库侧按全集统计施加
payload : {text, source_id, chunk_id, source}  # source_id/chunk_id 用于引用溯源
```

**检索**：dense + sparse 双路 `Prefetch`（各召回 `RECALL_K=20`）→ Qdrant **原生
`FusionQuery(RRF)`** 融合 → BGE `rerank` 取 `RERANK_TOP_K=3`。

```
query ──┬─ dense  Prefetch(limit=20) ─┐
        └─ sparse Prefetch(limit=20) ─┴─ RRF 融合 ─→ BGE 重排 ─→ top-k（带 score + 来源）
```

## 关键设计取舍

- **RRF 交给 Qdrant 库侧**，不手写融合：少一层易错代码，排名稳定（RRF 只看名次、对两路分数量纲不敏感）。
- **IDF modifier 放库侧**：BM25 的 IDF 需要**全集**词频统计；查询侧只产 term 权重，IDF 由 Qdrant
  按集合全量施加，避免查询侧拿不到全局统计。
- **重排是召回后的二次精排**：双路召回宽（各 20）保证不漏，reranker 在小候选集上精排保证准。
- **双模式零改动**：开发期本地落盘（`QDRANT_PATH`），部署期连服务器（`QDRANT_URL`），
  `vectordb.get_client()` 见 `QDRANT_URL` 即连服务器，否则落盘。切环境变量即可。

## 验证

- `eval_retrieval.py`：金标集（`gen_eval_set.py` 生成）算 **recall@k**——金标 chunk 是否被召回。
- `compare_retrieval.py`：混合检索 vs 纯 dense 的同查询对比，直观看 sparse 补了哪些专有名词命中。
- CLI 直查：`python kb_search.py "氨氯地平能降血压吗" --topk 5`。

## 待办

- 本地落盘模式 >2 万点会有性能告警（无真正 HNSW 索引）→ 服务器版解决（见 [P6](P6_DEPLOY.md)）。
- chunk 策略目前是通用递归切分；医疗教材的结构化切分（按「适应证/不良反应/禁忌」小节）可进一步提召回质量。

