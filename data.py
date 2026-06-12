"""HotpotQA 加载。

distractor 设定：每条样本自带 context（10 段文字 = 2 段金标 + 8 段干扰），
适合做 M1 基线——检索就是从这 10 段里挑相关的喂给模型，无需先建全库索引。

字段：
  question / answer / type / level
  supporting_facts: {title: [...], sent_id: [...]}
  context:          {title: [...], sentences: [[...], ...]}

注意：hotpot_qa 在 HF 上是脚本式数据集，新版 datasets 需要 trust_remote_code=True。
若仍报错，可固定 datasets 版本，或从官网下 JSON 后改成本地读取。
"""
from datasets import load_dataset


def load_hotpotqa(split: str = "validation", n: int | None = None):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if n is not None:
        ds = ds.select(range(n))
    return ds
