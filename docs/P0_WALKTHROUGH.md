# P0 完整操作清单：Qdrant 持久向量库 + 医疗 KB ingestion

> 提交：`014649e P0: Qdrant 持久向量库 + 医疗 KB ingestion 管道`

P0 的目标：**从零搭一个"持久化的医疗向量知识库"，并能往里灌数据、从里检索。**
之前的项目没有持久库（HotpotQA 每题临时给段落），P0 把这条腿立起来了。

---

## 1. 配置层（`config.py` 新增）

加了三组配置，全部可被 `.env` 覆盖：

```python
EMBED_DIM = 1024            # BGE-large-zh 的向量维度
QDRANT_PATH = "./qdrant_db" # 本地落盘目录（开发期）
QDRANT_URL = None           # 设了就连服务器（部署期）
QDRANT_COLLECTION = "medical_kb"  # 集合名（类似数据库的"表"）
CHUNK_SIZE = 400            # 每个切块最大字数
CHUNK_OVERLAP = 80          # 相邻切块的重叠字数
```

**关键设计**：Qdrant 同一套代码，开发用本地落盘（`path=`，无需 Docker），
部署只要设个 `QDRANT_URL` 就切到服务器——这就是为什么 Docker 没装也能先跑。

## 2. 依赖（`requirements.txt` 新增）

```
qdrant-client             # Qdrant 向量库客户端
langchain-text-splitters  # 切块工具
tqdm                      # 进度条
```

## 3. 灌库管道（`ingest.py` 新建）——P0 的核心

完整的 **ingestion 五步**，对应 JD 的"RAG 流程"：

```
① 读取    medical_book_zh.json（JSONL，8475 条医学教材段）
② 切块    每段用 RecursiveCharacterTextSplitter 重切到 ≤400 字
③ 嵌入    BGE 把每个 chunk 转成 1024 维向量（每批 64 个）
④ 入库    打包成 Qdrant 的 Point（向量 + 原文 + 元数据）写进集合
⑤ 计数    报告写入了多少
```

**② 为什么要重切**（核心知识点）：原始教材按 2048 字一段，但 **BGE 的输入上限是
512 token**（中文约 512 字）。直接嵌入 2048 字会被**截断**，后半段信息全丢。所以
必须切到 400 字以内。切的时候按"段落→句号→逗号"优先级切，尽量不切断语义，相邻块
留 80 字重叠防止答案正好被切在边界。

**④ 每个 Point 存了什么**（`payload` = 附带数据）：

```python
{
  "text": "这段教材原文",       # 检索回来要给模型看的内容
  "source_id": 第几条教材段,    # 溯源
  "chunk_id": 这段的第几个切块,
  "source": "medical_book_zh"   # 来源标记（将来融合多源时区分内部/外部）
}
```

向量用 **COSINE 余弦相似度**度量（文本检索标准做法）。

**命令行开关**：`--limit N`（只灌前 N 条，验证用）、`--recreate`（清空重建）。

## 4. 检索验证（`kb_search.py` 新建）

实现了 **两段式检索**（生产级 RAG 标配）：

```
问题 → BGE 向量化 → Qdrant 向量召回 top-20（快、粗）
                  → BGE 重排 top-3（慢、精）→ 返回
```

为什么两段：向量召回快但不够准，先粗筛 20 条；再用更强的重排模型精排出最相关的
3 条。这比单纯向量检索准很多。

## 5. 其他

- `.gitignore` 加了 `qdrant_db/`（落盘库可由 ingest 重建，不进 git）。
- 装了三个新依赖。

## 6. 实测验证结果

```
灌库：200 教材段 → 切成 1392 个 chunk → 全部嵌入入库
检索："临床药理学研究什么" → top-3 全是相关教材段，score 0.999/0.997/0.996
```

证明整条管道**从灌库到检索完全打通**。

---

## P0 给你带来的（简历 / JD 对应）

| P0 产出 | 对应 JD 要求 |
|---------|-------------|
| Qdrant 持久向量库 | ✅ 向量数据库 |
| ingestion 管道（解析→切块→嵌入→入库） | ✅ RAG 流程 + 文档处理 |
| 两段式检索（召回+重排） | ✅ 检索系统 |
| 本地/服务器双模式设计 | ✅ 为部署铺路 |

**一句话**：P0 把"临时段落检索"升级成了"**真·持久化向量知识库 + 完整灌库管道**"，
这是从"做过 RAG 作业"迈向"能搭 RAG 系统"的第一步。

---

## 相关命令速查

```bash
# 小批验证（前 200 条教材段）
python ingest.py --limit 200 --recreate

# 全量灌库（8475 段 → 约 5.9 万 chunk，约 75 分钟）
python ingest.py --recreate

# 检索验证
python kb_search.py "临床药理学主要研究什么"
python kb_search.py "高血压怎么治" --topk 5
```

## 下一步

- **全量灌库**：把 8475 段全部灌满（后台跑）。
- **P1**：把 agent 的检索从"HotpotQA 临时段落"改成"查 Qdrant 医疗库"，跑通第一个真·医疗 RAG。
