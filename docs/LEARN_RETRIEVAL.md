# 学习串讲：检索层 / Tool Use（混合检索）

> 整个 RAG 的地基，也是简历最硬的技术点。配合 `kb_search.py` / `embeddings.py` /
> `sparse.py` / `vectordb.py` / `ingest.py` 阅读。

## 0. 这一层解决什么

RAG 的 **R（Retrieval 检索）**。DeepSeek 不知道 5 万多段医疗教材的内容，靠这一层把相关段落
找回来喂给它。`retrieve_internal` 里 `search(sq)` 的背后就是这一层。

## Tool Use 全貌（先理清三个层次）

别把「模型」「工具」「支柱」混在一起：

```
支柱：Tool Use（Agent 会「调用工具」拿自己没有的信息）
  ├─ 工具①：内部知识库检索  ← 这里面才用到那几个模型
  │     ├─ bge-m3              （算 dense 语义向量）
  │     ├─ BM25               （算 sparse 词面向量）
  │     └─ bge-reranker-v2-m3 （重排；经 /v1/rerank 接口）
  └─ 工具②：外部 Web 检索  ← Tavily（external.py），这也是 Tool Use！
        ↓
      fuse 融合两路（知识融合）
```

- **模型 ≠ 工具 ≠ 支柱**：bge-m3/BM25/reranker 只是「内部检索」这**一个工具内部的零件**；
  「内部检索」「外部检索」是两个**工具**；「会调用这些工具」才是 **Tool Use 支柱**。
- 本讲后面主要拆「内部检索」（技术含量最高）；外部 Tavily 见文末附录，与它平级、同属 Tool Use。
- 深度点：严格的 Tool Use（function calling）是**让 LLM 自己决定调哪个工具**；本项目是图里
  **固定编排**调用检索（更接近 workflow 式工具使用），外部检索还带 `use_external` 开关由用户控制。
  这是可扩展方向。

## 1. 核心架构：两阶段（召回 → 重排）

```
5万+ 教材段
   │
① 召回 recall   ← 快速粗筛，从 5万 里捞回 20 条候选（要快、要广）  RECALL_K=20
   │
② 重排 rerank   ← 精细打分，从 20 条里挑最相关的 3 条（慢但准）   RERANK_TOP_K=3
   │
喂给 LLM 的 3 条证据
```

**为什么分两步？** 精排模型（cross-encoder）准但慢，对 5 万段逐个打分跑不动；先用快方法
粗筛到 20 条（召回），再用准方法精排这 20 条（重排）。兼顾速度和精度。

## 2. 召回：混合检索（dense + sparse 两路）

| 路 | 叫法 | 怎么找 | 擅长 | 短板 |
|----|------|--------|------|------|
| **dense** | 稠密 / 语义 | bge-m3 把文字变向量，按**语义相似**找 | 抓"意思相近"（"降血糖药"→"二甲双胍"）| 可能漏精确词 |
| **sparse** | 稀疏 / 词面 | BM25，按**词的精确匹配 + 词频**找 | 抓药名/术语**精确命中** | 不懂同义 |

**为什么两路都要？** 纯 dense 精确性不足（问"二甲双胍"会混回一堆别的降糖药）；纯 sparse
召回不到同义改写（问"降血糖的药"对不上"二甲双胍"）。互补：医疗里药名要精确（sparse 强），
也要懂同义（dense 强）。

## 3. RRF：怎么合并两路

两路各召回 20 条，但分数尺度不同（dense 是 cosine 0~1，sparse 是 BM25 几十），没法直接比。

**RRF（Reciprocal Rank Fusion，倒数排名融合）**：不看分数、看排名，每条最终分 =
`Σ 1/(k + 该路排名)`。两路都靠前的浮上来。好处：不用归一化不同尺度，简单鲁棒。Qdrant
原生支持（`FusionQuery(fusion=RRF)`）。

## 4. 重排：rerank（精排）

召回的 20 条用 BGE reranker 精打分取 top-3。和召回 dense 的区别：
- 召回用**双塔 bi-encoder**：query 和 doc **分别**编码再比相似度——快（doc 向量可预存），但粗。
- 重排用 **cross-encoder**：query 和 doc **拼一起**进模型出相关度——准（能看交互），但慢（每条现算）。
- 所以 cross-encoder 只在「已缩到 20 条」后用。

## 5. 对照 `kb_search.py`

```python
def hybrid_recall(query, recall=RECALL_K):       # ① 召回
    dense_q = get_embeddings().embed_query(query) # query → dense 向量
    sidx, sval = sparse.embed_query(query)        # query → sparse(BM25) 向量
    hits = client.query_points(
        prefetch=[                                # 两路并行召回
            models.Prefetch(query=dense_q, using="dense",  limit=recall),
            models.Prefetch(query=SparseVector(...), using="sparse", limit=recall),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # RRF 融合
        limit=recall,
    ).points
    return [{text, source_id, chunk_id} ...]      # 带来源元数据（供引用）

def search(query, recall=RECALL_K, top_k=RERANK_TOP_K):  # ② 召回+重排
    cands = hybrid_recall(query, recall)          # 先召回 20 条
    docs = [c["text"] for c in cands]
    ranked = rerank(query, docs, top_n=top_k)     # 再精排取 3 条
    return [{**cands[r["index"]], "score": ...} for r in ranked]
```

`search()` 是检索层对外总入口，`retrieve_internal` 调它。一行 `search(sq)` 背后是
「双路召回 → RRF → 重排」整套。

## 6. 配套文件

| 文件 | 干嘛 |
|------|------|
| `embeddings.py` | dense 向量化（`XinferenceEmbeddings`）+ rerank（打 `/v1/rerank` REST）|
| `sparse.py` | BM25 稀疏向量（FastEmbed 本地算）|
| `vectordb.py` | 连 Qdrant（本地落盘 / 服务器双模式）|
| `ingest.py` | 灌库（离线一次性）：每段切块后**同时算 dense + sparse 两个向量**存进 Qdrant |

**关键对称性**：灌库时每段存「dense+sparse」两个向量，检索时就用「dense+sparse」两路查——
存和查对称。这就是 `ingest.py` 一个 chunk 存两个向量的原因。

## 7. 完整例子

```
query = "二甲双胍的副作用"
  ├─ dense:  embed → 语义向量 → Qdrant 找语义相近 20 段
  ├─ sparse: BM25  → 词面向量 → Qdrant 找"二甲双胍""副作用"精确命中 20 段
  └─ RRF 融合两路 → 综合排名取 20 条候选
        → rerank（cross-encoder 精打分）→ 取最相关 3 条
        → 返回 3 条（带 source_id，供作答标 [引用]）
```

## 8. 面试问题清单

1. 为什么用混合检索，不直接向量检索？→ dense 抓语义、sparse 抓精确词面，互补。
2. dense 和 sparse 区别？→ 稠密语义向量 vs BM25 词面匹配。
3. RRF 是什么，为什么用它？→ 按排名倒数加权，免去归一化不同分数尺度。
4. 召回和重排为什么分两步？→ 召回快粗筛、重排准精排，兼顾速度精度。
5. 召回双塔 vs 重排 cross-encoder？→ 分别编码（快可预存）vs 拼一起编码（准但慢）。
6. 为什么 embedding/rerank 用 Xinference 不用 DeepSeek？→ DeepSeek 没 embedding 接口，BGE 在本地 GPU。
7. RECALL_K=20、RERANK_TOP_K=3 怎么定？→ 召回广不漏、重排精取少数喂 LLM，召回率与噪声的平衡。
8. 灌库为什么一个 chunk 存两个向量？→ 存查对称，两路要分别有 dense 和 sparse。

## 9. 自测题

1. 一句话说清「召回」和「重排」各自的目标和特点。
2. 为什么 dense 和 sparse 要一起用？各举一个对方搞不定的例子。
3. RRF 融合为什么"看排名"而不是"看分数"？
4. `search()` 内部依次做了哪几件事？

---

# 附：bge-m3 和 BM25 到底做了什么（向量化）

`ingest.py` 对**每段教材同时算两个向量**存进 Qdrant：

```python
embedder = get_embeddings()                      # bge-m3
for batch in 分批(chunks):
    dense_vecs  = embedder.embed_documents(texts) # ← bge-m3：算稠密向量
    sparse_vecs = sparse.embed_docs(texts)        # ← BM25：算稀疏向量
    points = [PointStruct(
        vector={"dense": dvec, "sparse": SparseVector(...)},  # 一段存两个向量
        payload={"text":..., "source_id":..., "chunk_id":...},
    ) ...]
    client.upsert(...)
```

两个「模型」性质完全不同：

## bge-m3 —— 稠密语义向量（dense）
- **是什么**：神经网络 embedding 模型（BAAI BGE 系列），跑在 Xinference（GPU）。
- **做什么**：把一段文字 → 1024 个浮点数（一个向量），把「语义」压进去。
  `"二甲双胍的副作用" → [0.12, -0.03, 0.87, ...]`
- **核心**：语义相近的文字向量距离近。"二甲双胍的副作用" 和 "服用甲福明后的不良反应"
  字面不同但意思一样 → 向量接近。检索用 cosine 余弦相似度比较。
- **「稠密」**：1024 维每一维都有值，是模型学出来的。这是传统意义的「向量化/embedding」。

## BM25 —— 稀疏词面向量（sparse）
- **是什么**：⚠️ 不是神经网络，是经典**统计算法**（搜索引擎经典打分公式），FastEmbed 本地算（CPU）。
- **做什么**：不理解语义，只看「词」。三要素：
  - **TF 词频**：词在这段出现越多越重要。
  - **IDF 逆文档频率**：词越罕见区分度越高（"的"没用，"二甲双胍"很有意义）。本项目 IDF 由 Qdrant 库侧施加（`modifier=IDF`）。
  - **长度归一化**：长文档不因词多占便宜。
- **输出**：稀疏向量 = 词→权重表 `{"二甲双胍": 2.3, "副作用": 1.8}`。
- **「稀疏」**：词表几万词，一段只含几个 → 向量绝大部分是 0。
- **核心**：精确匹配，不懂"甲福明≈二甲双胍"（这由 dense 来补）。

## 对比表

| | bge-m3（dense） | BM25（sparse） |
|--|--|--|
| 本质 | 神经网络模型 | 统计算法（非模型）|
| 在哪算 | Xinference（GPU）| FastEmbed 本地（CPU）|
| 向量长相 | 1024 维都有值（稠密）| 大部分是 0（稀疏）|
| 抓什么 | 语义（意思相近）| 词面（精确匹配+词频）|
| 懂同义吗 | 懂 | 不懂 |
| 擅长 | 同义改写、口语化提问 | 药名/术语精确命中 |

## 容易混的点（面试加分）
严格说只有 **bge-m3 是真正的「向量化模型」**，**BM25 是「统计打分算法」**——只是在 Qdrant
混合检索框架里它的结果也被表示成「稀疏向量」，所以统称两路向量。存查对称：灌库存两个向量，
检索才能两路分别查再 RRF 融合。

---

# 附：召回为什么「快」（预存 + 索引）

召回要从 5 万多段里找最相似的 20 段。快的根子是**避免逐个比对**，靠两招：

## 朴素做法为什么慢
暴力检索：query 向量和库里 5 万个向量逐个算 cosine，排序取前 20。复杂度 O(N)，百万级就卡死。
召回的「快」就是要绕开它。

## 快的关键①：向量预先算好存库
dense 向量在 `ingest.py` **灌库时就算好存进 Qdrant**，不是查询时才算。所以查询时**只算 1 个
query 向量**，库里 5 万个早备好了。
> 对比 rerank 为什么慢：cross-encoder 必须把 query 和每个候选**拼一起现算**，没法预存，
> 20 条跑 20 次模型。能不能预存，是召回快、重排慢的根本。

## 快的关键②：索引（不扫全库，跳着找）
- **dense 路 → HNSW 索引（近似最近邻 ANN）**：把向量组织成一张图，查询时从入口**跳着找邻居**
  （像社交网络找人，几跳就到），不问遍所有人。近似最近邻，牺牲一点点准确率换几个数量级速度
  （O(N) → 约 O(log N)）。
- **sparse 路 → 倒排索引**：预建「词 → 含该词的文档列表」表，查"二甲双胍"只看含这个词的文档，
  不碰其它几万段。搜索引擎几十年的看家本领。

## 对照代码
灌库建索引（`ingest.py` `ensure_collection`）：
```python
client.create_collection(
    vectors_config={"dense": VectorParams(size=1024, distance=COSINE)},  # → 自动建 HNSW
    sparse_vectors_config={"sparse": SparseVectorParams(modifier=IDF)},  # → 倒排索引
)
```
查询用索引召回（`kb_search.py` `query_points(prefetch=[dense, sparse], limit=20)`）：Qdrant 内部
用索引秒级返回，不暴力扫全库。

## 回扣部署
`P6_DEPLOY.md` 说"qdrant 服务器版有真正的 HNSW 索引，解决本地落盘模式 >2万点性能告警"——
说的就是这里的索引。本地落盘 HNSW 支持弱，点一多召回就慢；服务器版才有完整 HNSW。

## 缩写怎么念（口语/面试）
- **HNSW** → 念字母 H-N-S-W；可补全称 Hierarchical Navigable Small World（分层可导航小世界图）。
- **ANN** → 念字母（近似最近邻）；**RRF / BGE / TF-IDF** → 念字母；**BM25** → 念 B-M-25。
- **RAG** → 例外，当单词念「拉格 /ræɡ/」。
- 口诀：大多数缩写念字母，只有 RAG 习惯当单词念。

---

# 附：RRF 算例 + rerank 操作演示

## RRF 怎么合并两路（算一遍）

query「二甲双胍的副作用」，两路各召回前 4 名：

- dense（语义）排名：1=A, 2=B, 3=C, 4=D
- sparse（词面）排名：1=C, 2=A, 3=E, 4=B

RRF 公式：每个文档总分 = 各路 `1/(k+排名)` 相加（k 默认 60）。用最直观的 `1/排名` 看：

| 文档 | dense 贡献 | sparse 贡献 | 总分 | 说明 |
|------|----------|-----------|------|------|
| **A** | 1/1=1.0 | 1/2=0.5 | **1.5** | 两路都靠前 → 最高 |
| **C** | 1/3≈0.33 | 1/1=1.0 | **1.33** | 两路都靠前 → 第二 |
| **B** | 1/2=0.5 | 1/4=0.25 | **0.75** | 两路中等 |
| **E** | 没出现=0 | 1/3≈0.33 | **0.33** | 只一路认可 → 靠后 |
| **D** | 1/4=0.25 | 没出现=0 | **0.25** | 只一路认可 → 最后 |

**融合后：A > C > B > E > D**

RRF 的灵魂：
1. 两路都认可的（A、C）浮到最前。
2. 只一路出现的（D、E）靠后。
3. **全程只用「排名」不用原始分数** → 不用管 dense（cosine 0~1）和 sparse（BM25 几十）尺度不同。
4. 真实公式 `1/(k+排名)`，k=60，作用是让名次更平滑、防止第 1 名碾压。顺序仍是 A>C>B>E>D。

## rerank（精排）怎么操作

承接：RRF 后有 20 条候选，rerank 重新精打分取最好的 3 条。

步骤：
1. 把 query 和 20 条文本一起发给 BGE reranker（打 `/v1/rerank`）。
2. cross-encoder 对每条把 `(query, 候选文本)` 拼一对送进模型，输出 `relevance_score`。
3. 按分降序取 top_n=3。
4. 返回 `[{index, relevance_score}]`，index 是输入下标。

```python
def rerank(query, documents, top_n):
    payload = {"model": RERANK_MODEL_UID, "query": query, "documents": documents}
    resp = requests.post(f"{XINFERENCE_URL}/v1/rerank", json=payload)
    return resp.json()["results"]   # [{index, relevance_score, ...}]

# kb_search.search 里：
ranked = rerank(query, docs, top_n=3)
return [{**cands[r["index"]], "score": r["relevance_score"]} for r in ranked]
#            ↑ 用 index 映射回原候选，拿回 source_id 等元数据
```

直观例子（rerank 的"精"）——query「二甲双胍的副作用」：

| 候选 | 内容 | 召回阶段 | rerank |
|------|------|---------|--------|
| 候选1 | "二甲双胍常见副作用有恶心、腹泻…" | 排得还行 | **0.95** ✓ |
| 候选2 | "二甲双胍的**作用机制**是…" | 含"二甲双胍"排不低 | **0.30** |
| 候选3 | "**格列美脲**的副作用有…" | 含"副作用"排不低 | **0.20** |

候选2/3 召回时因部分匹配排不低，rerank 的 cross-encoder 能同时看懂"既要二甲双胍、又要副作用"，
把全中的候选1 顶到第一——纠正召回的"沾边但不精准"。

为什么 rerank 更准：召回 dense 是双塔（query/doc 分别编码，只比向量距离，没看交互）；rerank 是
cross-encoder（拼一起进模型，逐词看对应关系），更准但每条现算、只用于 20 条。

## 一句话串
两路各召回20 → RRF 按排名融合（两路都靠前的浮上来）→ 20候选 → rerank cross-encoder 精打分
→ 取最相关 3 条 → 喂 LLM。RRF 解决"怎么公平合并两路"，rerank 解决"怎么挑出真正最相关的"。

---

# 附：外部检索 external.py（Tool Use 的第二个工具）

内部检索（上面那套）和**外部 Web 检索**平级，**同属 Tool Use 支柱**。

## 工具是什么
**Tavily**——一个专为 LLM 优化的搜索 API（返回干净的正文，不用自己爬网页解析）。

## 干什么
补内部教材**没有**的：最新指南、时效信息、教材外的内容。内部教材是稳定基础知识，外部 Web
补"新"和"广"。

## 代码（external.py）
```python
from tavily import TavilyClient
def web_search(query, max_results=TAVILY_MAX_RESULTS):
    resp = _get_client().search(query=query, max_results=max_results)
    return [{"text": r["content"], "title": r["title"], "url": r["url"]} for r in resp["results"]]
```

## 节点（med_nodes.retrieve_external）的三个要点
```python
def retrieve_external(state):
    if not state.get("use_external"):        # ① 用户开关：默认关（快、省钱）
        return {"external_evidence": []}
    try:
        hits = web_search(q)
    except Exception:                        # ② 优雅降级：Tavily 挂了仅用内部源，不崩
        return {"external_evidence": []}
    return {"external_evidence": [{"text": h["text"][:800], ...} for h in hits]}  # ③ 截断 800 字
```
1. **`use_external` 开关**：用户控制要不要联网（前端那个 🌐 复选框）。默认关——仅查内部教材，更快更省。
2. **优雅降级**：外部 API 失败用 try/except 兜住，返回空、仅用内部源，服务不崩。
3. **截断 800 字**：网页正文常很长，截断避免拖慢重排、且对作答无必要。

## 和内部检索怎么合并
两路证据都进 `fuse`，对原问题**统一重排**取 top-k——最相关的浮上来，不论来自教材还是网页。
这就是「知识融合」。

## 一句话
Tool Use 支柱 = 内部检索工具（bge-m3+BM25+reranker 查 Qdrant）+ 外部检索工具（Tavily 查网络），
`fuse` 融合两路。外部检索是更"标准"的 Tool Use（调外部 API），带用户开关和优雅降级。
