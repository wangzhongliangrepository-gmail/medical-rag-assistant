# CLAUDE.md — 个人学习助手 Agent

给 Claude Code 的项目上下文。开始任何任务前先读本文件，并遵循其中的约束。

## 项目目标

用 LangGraph 构建一个「个人学习助手 Agent」：用户提一个（可能要跨多段资料综合的）问题，Agent 规划检索步骤、调检索工具找证据、用记忆积累上下文与经验、用反思判断证据是否充分并改写，最后给出带引用的回答。

简历定位：一个同时落地 Agent 四大支柱（Memory / Planning / Tool Use / Reflection）、且有客观评测指标的项目。叙事核心是「基线 + 逐层改进 + 消融」——每加一个支柱，报告一次 EM/F1 增量。

## 四大支柱 → LangGraph 映射

- **Planning**：plan-and-execute。planner 节点把问题拆成子问题 / 检索计划。
- **Tool Use**：检索工具（BGE 向量检索 + 重排）。
- **Memory**：
  - 短期 / 工作记忆：线程内 state，靠 checkpointer 持久化。
  - 长期记忆：跨线程 Store（带 BGE 语义检索），两个 namespace——事实缓存 + 经验 / 教训库。
- **Reflection**：Reflexion 式。reflect 节点批判当前证据与草稿（够不够、有无臆造），不够则回到 planner / retrieve，带修订次数上限。

图主干：`question → planner → retrieve → draft → reflect →（够了→finalize / 不够→planner，≤N 轮）→ distill（写长期记忆）→ 输出答案+引用`。

## 技术栈与硬约束

**LLM：DeepSeek V4 云端 API，经 langchain-deepseek 的 ChatDeepSeek 接入。**
- 默认 `deepseek-v4-flash`（快、便宜，支持工具调用与结构化输出）。难节点（planner、reflect）可选 `deepseek-v4-pro`。
- 不要用 `deepseek-chat` / `deepseek-reasoner`——这两个旧别名 2026-07-24 停用。
- DeepSeek 思考内容在独立的 reasoning_content 字段，`.content` 是干净的，不需要剥 `<think>`。
- 上下文缓存默认开启：把稳定的系统提示和重复的检索上下文放在 prompt 前部并保持一致，可命中缓存降本——这对 RAG/agent 反复重发上下文很有用。

**向量化 + 重排：Xinference 上的 BGE（本地，经 SSH 隧道 9997）。**
- DeepSeek 没有 embedding 接口，所以这部分留在 Xinference，别试图用 DeepSeek 做向量化。
- `XinferenceEmbeddings` 用 model_uid（不是模型名）。
- 重排没有现成 LangChain 封装，打 `/v1/rerank` REST（见 `embeddings.py`），不要自己造别的。

**编排：LangGraph。** StateGraph + 条件边实现反思回路；checkpointer 管短期记忆，Store 管长期记忆。

**评测数据：HotpotQA distractor（datasets 加载）。** 每条自带 10 段 context（2 段金标 + 8 段干扰）；做 benchmark 时直接从这 10 段里检索，不必先建全库索引。指标：EM / F1，按 HotpotQA 官方口径做答案归一化（去标点、去冠词、小写、空白规整）。

## 目录

```
config.py        端点 / 模型 / UID（.env 覆盖，唯一配置入口）
llm.py           ChatDeepSeek 封装
embeddings.py    Xinference 向量化 + 重排
data.py          HotpotQA 加载
smoke_test.py    M0 冒烟测试
# 待建：
state.py         图状态 TypedDict
nodes.py         各节点函数
graph.py         组装 StateGraph + compile
eval.py          在 HotpotQA 上跑 EM / F1
```

## 里程碑

- **M0 脚手架**：四条管道 + LangGraph 冒烟测试（已完成，跑 `python smoke_test.py` 验证）。
- **M1 基线**：单跳 RAG。10 段 context 用 BGE 重排取 top-k → DeepSeek 直接答 → 在 validation 上算 EM/F1。这是对照基线。
- **M2 +Planning**：plan-and-execute 多跳拆解 → 报 EM/F1 增量。
- **M3 +Reflection**：Reflexion 回路 → 报增量。
- **M4 +Memory**：短期 checkpointer + 长期 Store（事实缓存 + 经验记忆）→ 报准确率抬升 + 检索次数下降。
- 全程接 LangSmith 追踪。

## 约定

- 写代码前先读相关已有文件；所有端点 / 模型 / 密钥只从 `config.py` 取，不要在别处硬编码。
- 密钥只放 `.env`，绝不写进代码、日志或提交。
- 每个里程碑：先实现，再用 `eval.py` 量化，把指标填进 `README.md` 的结果表；改进一律对照基线报增量。
- 解析模型输出时优先用结构化输出（DeepSeek 支持 JSON / 工具调用），不要用脆弱的正则去抠。
- 改完跑一遍相关脚本验证再说「完成」。
