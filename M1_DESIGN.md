# M1 基线设计决策

## top-k 取多少？

取 **top-3**。

HotpotQA 金标是 2 段，top-3 保证大概率覆盖到，又不会让 prompt 太长。
后续做消融时可对比 top-2 / top-5 的 EM/F1 差异。

## context 格式化

带编号和标题喂给 DeepSeek：

```
[1] 标题：xxx
内容：yyy

[2] 标题：xxx
内容：yyy
```

比裸文本更容易让模型定位答案，同时为后续加引用（M3+）打基础。

## Prompt

```
根据以下资料，简洁地回答问题。只输出答案本身，不要解释。

{context}

问题：{question}
答案：
```

"只输出答案本身"这句关键——HotpotQA 的 gold answer 都是短语，
模型若输出完整句子会拉低 F1。

## eval 规模

| 阶段 | 样本量 | 用途 |
|------|--------|------|
| 快速验证 | 200 条 | 每次迭代调参 |
| 最终对比 | 7405 条（全量） | 里程碑 EM/F1 汇报 |

200 条约 2-3 分钟，够统计意义也够快。

## 检索方式

直接对每题的 10 段调 BGE `/v1/rerank` 打分，不需要先 embed question 再向量召回。
M1 逻辑更干净，少一个可能出错的环节。向量召回从 M2 开始按需引入。

## 数据流

```
HotpotQA 样本（10 段 context）
  → 每段拼成纯文本（title + sentences）
  → BGE reranker 对 10 段打分排序
  → 取 top-3
  → DeepSeek：question + top-3 → 答案
  → 与 gold answer 算 EM / F1
```

## 评测指标口径

按 HotpotQA 官方做答案归一化：

1. 小写
2. 去标点
3. 去冠词（a / an / the）
4. 规整空白

- **EM**：归一化后完全匹配得 1，否则 0
- **F1**：词级别的 precision / recall 调和平均