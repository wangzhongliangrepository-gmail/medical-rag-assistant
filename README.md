# 个人学习助手 Agent — M0（脚手架，DeepSeek 版）

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

## 结果表（随里程碑填）

| 阶段 | 配置 | EM | F1 |
|------|------|----|----|
| M1 | 单跳 RAG 基线（top-3，deepseek-v4-flash，n=200） | 0.5550 | 0.6971 |
| M1 bridge | └─ bridge 子集（n=166） | 0.5241 | 0.6759 |
| M1 comparison | └─ comparison 子集（n=34） | 0.7059 | 0.8006 |
| M2-static | +Planning 静态子问题（n=200） | 0.5250 | 0.6656 |
| M2-static bridge | └─ bridge 子集（n=166） | 0.5241 | 0.6735 |
| M2-static comparison | └─ comparison 子集（n=34） | 0.5294 | 0.6269 |
| M2-refine | +Planning 链式精化（n=200） | 0.5650 | 0.7024 |
| M2-refine bridge | └─ bridge 子集（n=166） | 0.5602 | 0.7065 |
| M2-refine comparison | └─ comparison 子集（n=34） | 0.5882 | 0.6823 |
| M3 | +Reflection | | |
| M4 | +Memory | | |

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

## 下一步（M3：+Reflection）

M2-refine 验证了链式精化对 bridge 有效（+0.036 EM），comparison 仍低于 M1 基线。
M3 加入 Reflexion 回路，让模型自判证据是否充分，不够则重检索，上限 N 轮。
