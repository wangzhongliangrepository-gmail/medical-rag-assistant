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
| M1 | 单跳 RAG 基线（top-3，deepseek-v4-flash，n=200） | 0.5250 | 0.6503 |
| M2 | +Planning | | |
| M3 | +Reflection | | |
| M4 | +Memory | | |

## 下一步（M1：单跳 RAG 基线）

把 HotpotQA 每条的 10 段 context 用 BGE 重排取 top-k，喂给 DeepSeek 直接作答；在 validation 上算 EM/F1，作为后面加 Planning / Reflection / Memory 的对照基线。
