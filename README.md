# 个人学习助手 Agent（DeepSeek 版）

进度：M0 脚手架 → M1 单跳基线 → M2 +Planning → **M3 +Reflection（当前）** → M4 +Memory（待做）。
四大支柱：Planning / Tool Use / Reflection / Memory，HotpotQA distractor 上以 EM/F1 量化逐层增量。

目标：在写 agent 逻辑之前，先把四条管道打通——DeepSeek 能应答、BGE 能向量化、BGE 能重排、HotpotQA 能加载，外加确认 LangGraph 装好。

LLM 改用 DeepSeek 云端 API（不再跑本地 Qwen3）。向量化和重排仍走 Xinference 上的 BGE，因为 DeepSeek API 不提供 embedding 接口。

## 目录结构

```
learning-assistant/
├── CLAUDE.md        # 给 Claude Code 的项目上下文，开工前先读
├── requirements.txt
├── .env.example
├── config.py        # 端点 / 模型 / UID（可被 .env 覆盖）
├── llm.py           # ChatDeepSeek 封装
├── embeddings.py    # Xinference 向量化 + 重排（REST）
├── data.py          # HotpotQA 加载
└── smoke_test.py    # 四条管道 + LangGraph 的冒烟测试
```

## 前置条件

1. 去 platform.deepseek.com 申请 API 密钥，填进 `.env` 的 `DEEPSEEK_API_KEY`。
2. SSH 隧道：本机 `9997` → AutoDL 的 Xinference（LLM 走云端，所以只剩这一条隧道）。
3. Xinference 已启动 BGE 向量模型与 BGE 重排模型，用 `xinference list` 记下两者的 **model UID**。

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env          # 填 DEEPSEEK_API_KEY，并把两个 *_MODEL_UID 改成你实际启动的 UID
python smoke_test.py
```

## 成功长这样

```
[LLM] DeepSeek -> 向量检索是把文本转成向量后按相似度找最接近的内容……
[EMBED] BGE 维度=1024  前三维=[...]
[RERANK] 最相关 -> 过拟合指模型在训练集上表现很好但泛化能力差。  分数=0.99xx
[DATA] HotpotQA 已加载  Q: ...
        A: ...  context 段数: 10  金标条数: 2
[GRAPH] LangGraph OK，1+1 -> 2
```

## 结果表（n=600，本地 Xinference 同后端，deepseek-v4-flash，temp=0）

> 同一批 600 条 validation、同一本地后端测得，可横向对比。
> ⚠️ **评测噪声**：DeepSeek 云端 API 即使 `temperature=0` 仍非确定性（MoE 路由 + 服务端 batch）。
> 实证：M3 与"去 reflect"两组的 comparison 走**字节级相同**的代码路径，结果却差 0.026（≈3 题）。
> 故噪声带 ≈ **±0.02~0.03（n=114）/ ±0.015（n=486）**，**小于 ~0.03 的差异单次运行不可信**。

| 配置 | classify | Planning | Reflection | EM | F1 |
|------|:---:|:---:|:---:|----|----|
| **M1** 单跳基线 | ✗ | ✗ | ✗ | **0.5783** | 0.7153 |
| └─ bridge (n=486) | | | | 0.5720 | 0.7165 |
| └─ comparison (n=114) | | | | 0.6053 | 0.7105 |
| **M2-refine** 链式精化 | ✗ | ✓ | ✗ | **0.5900** | 0.7336 |
| └─ bridge (n=486) | | | | 0.6029 | 0.7503 |
| └─ comparison (n=114) | | | | 0.5351 | 0.6625 |
| **M3** +Reflection | ✓ | ✓ | ✓ | **0.5883** | 0.7257 |
| └─ bridge (n=483) | | | | 0.5942 | 0.7351 |
| └─ comparison (n=114) | | | | 0.5789 | 0.7049 |
| **M3 消融**：去 reflect | ✓ | ✓ | ✗ | 0.5733 | 0.7121 |
| └─ bridge (n=486) | | | | 0.5658 | 0.7093 |
| └─ comparison (n=114) | | | | 0.6053 | 0.7239 |
| **M4** +Memory | | | | 待填 | 待填 |

> 早期 n=200（旧 SSH 后端）的 M2-static 实验见下方「M2 消融分析」，后端/样本量不同，仅作定性参考。

### 逐层拆解（每次只动一个变量）

| 改动 | bridge | comparison | total | 解读 |
|------|--------|------------|-------|------|
| **+Planning**（M1→M2） | **+0.031** | **−0.070** | +0.012 | 多跳拆解抬 bridge；但拆解 comparison 反伤 |
| **+classify**（M2→M3去reflect） | −0.037 | **+0.070** | −0.017 | 路由救回 comparison；~4% bridge 误判被踢出 planning，反伤 |
| **+reflect**（M3去reflect→M3） | +0.028 | −0.026* | +0.015 | reflect 补回 bridge（*comparison 路径相同，此差为噪声） |

可信度分级：
- ✅ **铁证（远超噪声）**：拆解 comparison 反伤（−0.070）、classify 救回 comparison（+0.070）。
- ⚠️ **大概率真（略超噪声 + 有机制解释）**：Planning 抬 bridge（+0.031）、classify 误判伤 bridge（−0.037）。
- ❓ **噪声边缘（单次不下定论）**：reflect 抬 bridge（+0.028）。需多种子去噪才能钉死。

## M2 消融分析

### 静态子问题（M2-static）为何未能超越 M1

planner 一次性生成所有子问题，子问题之间没有依赖关系：

```
SubQ1: "Who portrayed Corliss Archer in Kiss and Tell?"
SubQ2: "What government position was held by the actress who played Corliss Archer?"  ← 仍是泛指
```

retrieve 只是换了查询词，证据池还是同样的 10 段文档，SubQ2 没有用 SubQ1 检索出的答案来精化查询，导致结果与 M1 持平。comparison 子集下跌明显（-0.18 EM），因为 M1 一次喂入所有上下文更利于跨实体对比。

### 链式精化（M2-refine）的改进

在 retrieve 和下一轮 retrieve 之间插入 refine 节点，提取上一跳的中间答案并代入下一个子问题：

```
SubQ1 检索 → refine 提取中间答案 "Shirley Temple"
→ SubQ2 精化为 "What government position did Shirley Temple hold?"
→ 再检索 → 答案更精准
```

## M3 消融分析（+Reflection / +classify）

M3 在 M2-refine 上加了两样东西：**classify 分流**（comparison 走 M1 短路径、bridge 走完整链）和 **Reflexion 反思回路**（答完自判证据是否充分，不够则重检索，上限 `MAX_REVISIONS=3`）。n=600 同后端消融结论：

### classify：修复 comparison 的双刃剑

M2-refine 把 comparison 也硬拆成子问题，跨实体对比被打散，comparison 暴跌（0.6053→0.5351，−0.070）。classify 用一个轻量分类节点（实测准确率 96%）把 comparison 路由回 M1 式一次性作答路径，**救回 +0.070**。代价：~4% 的 bridge 被误判成 comparison、踢出 planning 路径，**反伤 bridge −0.037**。净效果在 total 上接近持平。

### Reflection：边缘正贡献，但烧检索预算

固定 classify，只切 reflect 开关：bridge 0.5658（去 reflect）→ 0.5942（开 reflect），**+0.028**。方向为正，但**卡在噪声带边缘**（n=486 噪声 ≈±0.015，此差仅约 2 倍噪声），且 `avg_revisions=0.83` 意味着平均每题多烧 0.83 次检索。**这是"准确率换效率"的权衡，单次运行不足以下定论**——要钉死需多种子取均值。

> 注：早期在 n=200 噪声下曾误判"reflect 是纯负担"，n=600 推翻了该结论。这本身是教训——**小样本 + 云端 LLM 非确定性下，细微 EM 差异不可信**。

### 方法论：云端 LLM 的评测噪声

最关键的发现：`temperature=0` 下，M3 与"去 reflect"两组的 **comparison 走字节级相同的代码路径**，EM 却差 0.026（≈3 题）。说明 DeepSeek 云端 API 即使贪心解码也非确定性。**所有 <0.03 的单次 EM 差异都落在噪声里**，做消融时只能对大效应（拆解 comparison ±0.07 这类）下硬结论。

## 当前最优配置

n=600 total EM：M2-refine（0.5900）≈ M3（0.5883）> M1（0.5783）> M3-去reflect（0.5733）。**M2-refine 与 M3 在噪声内打平**，没有单一配置全面占优——这是 bridge↔comparison 的权衡前沿，而非单调增量。诚实结论：**Planning 是确凿赢家（抬多跳），classify 是对 comparison 回归的有效工程修复，Reflection 收益在噪声边缘。**

## 下一步（M4：+Memory）

见 `M3_STRUCTURE.md` 与设计讨论：HotpotQA 单轮独立问答对记忆不友好，M4 需为 Memory 配"对的尺子"（多轮指代消解 / 跨会话事实召回），而非硬套 HotpotQA EM/F1。
