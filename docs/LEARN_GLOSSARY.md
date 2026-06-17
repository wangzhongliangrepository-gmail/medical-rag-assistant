# 学习串讲：项目英文术语词表

> 项目里反复出现的英文「行话」。含大致读音 + 中文 + 在本项目里的含义。
> 面试里直接用英文术语反而显专业（如 conditional edge、retrieval、reranker）。

## 一、LangGraph 编排相关

| 英文 | 读音（大致） | 中文 | 在本项目里 |
|------|------------|------|-----------|
| **graph** | 格拉夫 | 图 | 把节点连成的流程图（StateGraph）|
| **state** | 斯戴特 | 状态 | 节点间传递的数据（`MedState`）|
| **node** | 弄得 | 节点 | 图里的一个处理步骤（一个函数）|
| **edge** | 埃奇 | 边、连线 | 连接两个节点的线 |
| **conditional** | 肯迪舍纳 | 有条件的 | 看情况判断（来自 condition 条件）|
| **conditional edges** | — | 条件边 | 看条件决定走哪条岔路（如 reflect 后分流）|
| **compile** | 肯派欧 | 编译 | 把图纸变成可执行对象 |
| **checkpointer** | 切克破因特 | 检查点存储器 | 短期记忆：按 thread_id 存会话 state |
| **store** | 斯多 | 仓库、存储 | 长期记忆：按 user_id 存用户健康事实 |
| **reducer** | 瑞丢瑟 | 归并器 | 决定 state 字段怎么合并（覆盖 or 累加）|

## 二、RAG / 检索相关

| 英文 | 读音 | 中文 | 在本项目里 |
|------|------|------|-----------|
| **retrieve / retrieval** | 瑞特里夫 | 检索、取回 | 把相关资料找回来（RAG 的 R）|
| **augment** | 奥格门特 | 增补、扩充 | 反思后再补一次检索增加证据 |
| **augment_retrieve** | — | 补充检索 | 证据不够 → 补检索把证据增补进来 |
| **embedding** | 因贝丁 | 嵌入、向量化 | 把文字变成向量（BGE 做）|
| **dense** | 丹斯 | 稠密（向量）| 语义向量检索那一路 |
| **sparse** | 斯帕斯 | 稀疏（向量）| BM25 词面检索那一路 |
| **rerank / reranker** | 瑞兰克 | 重排、重排器 | 对召回结果再精细打分排序 |
| **fuse / fusion** | 否兹 / 福订 | 融合 | 把内外部证据合在一起统一重排 |
| **RRF** | — | 倒数排名融合 | Reciprocal Rank Fusion，合并 dense+sparse 两路 |
| **recall** | 瑞考 | 召回 | 第一步粗筛捞回候选（recall@k 也是评测指标）|
| **chunk** | 强克 | 切块、文本块 | 教材切成的小段 |
| **ingest / ingestion** | 因杰斯特 | 灌库、摄入 | 把教材切块+向量化写进 Qdrant |

## 三、Agent 四大支柱相关

| 英文 | 读音 | 中文 | 在本项目里 |
|------|------|------|-----------|
| **planning** | 普兰宁 | 规划 | plan 节点拆解问题 |
| **tool use** | 图尔 尤斯 | 工具调用 | 检索、联网搜索 |
| **reflection / reflect** | 瑞弗莱克申 | 反思 | reflect 节点自检证据够不够 |
| **memory** | 麦默瑞 | 记忆 | 短期会话 + 长期用户记忆 |
| **contextualize** | 肯泰克斯求莱兹 | 结合上下文（改写）| 用历史把「它」改成具体实体（指代消解）|
| **standalone** | 斯坦德欧隆 | 独立的、自包含的 | standalone_question = 不依赖上下文的完整问题 |
| **revision(s)** | 瑞维任 | 修订（次数）| 反思重试了几轮 |
| **coreference** | 寇瑞菲伦斯 | 指代 | 指代消解 = coreference resolution |

## 四、工程 / 部署相关

| 英文 | 读音 | 中文 | 在本项目里 |
|------|------|------|-----------|
| **namespace** | 内姆斯佩斯 | 命名空间 | 长期记忆按 user_id 分隔的「格子」|
| **thread_id** | 思瑞德 | 线程/会话 id | 一次对话的唯一编号（短期记忆的钥匙）|
| **structured output** | 斯特拉克秋德 | 结构化输出 | 让 LLM 直接吐合法 JSON，不用正则 |
| **fallback** | 否拜克 | 兜底、降级 | 出错时的退路（如 Tavily 挂了仅用内部源）|
| **graceful degradation** | 格瑞斯否 | 优雅降级 | 部分失败不崩，能力下降但服务可用 |
| **volume** | 沃留姆 | 数据卷 | Docker 里持久化数据的磁盘（重启不丢）|
| **checkpoint** | 切克破因特 | 检查点 | 存档点，记录某时刻的状态 |

## 五、几个连起来记的组合词

- **augment_retrieve** = augment（增补）+ retrieve（检索）= 补充检索
- **conditional edges** = conditional（看条件）+ edges（边）= 条件边（岔路口）
- **coreference resolution** = 指代消解（把「它」还原成具体实体）
- **information retrieval** = 信息检索（检索这个领域的标准叫法）
- **reciprocal rank fusion (RRF)** = 倒数排名融合（合并多路检索结果）
