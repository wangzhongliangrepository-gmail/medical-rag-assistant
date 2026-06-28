"""检索评测集「反向生成」工具：从 Qdrant 库里抽 chunk → 让 LLM 据其生成问题 → 产出金标草稿。

思路：评测要「问题 → 金标 chunk」，正向人工标注贵；这里反过来——挑一个 chunk，让 LLM 生成
「这段内容能回答的问题」，那这个 chunk 天然就是该问题的金标。批量生成后**人工审核**即可。

产出 JSON（供人工审核后用于 eval_retrieval.py）：
  [
    {"question": "...", "gold": [[source_id, chunk_id]], "_chunk_text": "（原文，供审核）"},
    ...
  ]

⚠️ 这是「草稿」：① 问题是否自然合理、② 金标是否准确（可能还有别的 chunk 也该算金标）
   都需要你过一遍。审核时可手动往 gold 里补别的 [source_id, chunk_id]。

用法：
  python gen_eval_set.py --n 50                      # 随机抽 50 个 chunk 生成
  python gen_eval_set.py --n 50 --min-len 120        # 只用 ≥120 字的 chunk（信息更足）
  python gen_eval_set.py --n 50 --out my_eval.json
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse
import json
import random

from pydantic import BaseModel

from config import LLM_FLASH, QDRANT_COLLECTION
from llm import get_llm
from vectordb import get_client

_GEN_PROMPT = """你是医疗题库命题人。下面是一段中文医学教材。请提出**一个**患者或医学生
可能会自然问出、且**这段内容能够回答**的中文问题。要求：
- 口语化、像真实提问，不要照抄原文句子。
- 问题要具体（含具体药名/疾病/方面），不要空泛（如"这讲了什么"）。
- 只针对这段能回答的内容提问，不要问段落里没有的信息。

教材段落：
{text}"""


class GenQuestion(BaseModel):
    question: str


def iter_chunks(client):
    """翻页遍历整个集合，产出每个 chunk 的 payload（不取向量，快）。"""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            yield p.payload
        if offset is None:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="生成多少条评测样本")
    parser.add_argument("--min-len", type=int, default=80, help="只用长度 ≥ 此值的 chunk")
    parser.add_argument("--out", default="eval_set_draft.json", help="输出文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（可复现抽样）")
    args = parser.parse_args()

    client = get_client()
    print(f"[1/3] 遍历集合 '{QDRANT_COLLECTION}' 收集 chunk...")
    pool = [
        p for p in iter_chunks(client)
        if p.get("text") and len(p["text"]) >= args.min_len
    ]
    if not pool:
        print("（库为空或没有够长的 chunk：先跑 python ingest.py --recreate）")
        return
    print(f"      可用 chunk {len(pool)} 个（≥{args.min_len} 字）")

    random.seed(args.seed)
    sample = random.sample(pool, min(args.n, len(pool)))
    print(f"[2/3] 抽样 {len(sample)} 个，调 LLM 反向生成问题...")

    llm = get_llm(model=LLM_FLASH)
    structured = llm.with_structured_output(GenQuestion, method="json_mode")

    rows = []
    for i, p in enumerate(sample, 1):
        try:
            r = structured.invoke([
                ("system", '你是医疗命题人，只输出合法 JSON，格式：{"question": "..."}'),
                ("human", _GEN_PROMPT.format(text=p["text"])),
            ])
            q = (r.question or "").strip()
        except Exception as e:
            print(f"   [{i}/{len(sample)}] 跳过（生成失败：{type(e).__name__}）")
            continue
        if not q:
            continue
        rows.append({
            "question": q,
            "gold": [[p["source_id"], p["chunk_id"]]],   # 这段就是金标；审核时可再补
            "_chunk_text": p["text"][:400],              # 供人工审核对照（下划线前缀=元数据）
        })
        if i % 10 == 0:
            print(f"   已生成 {i}/{len(sample)}...")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[3/3] 完成：写入 {len(rows)} 条 → {args.out}")
    print("      ⚠️ 这是草稿，请人工审核：问题是否合理、gold 是否需要补别的 chunk。")
    print("      审核后用：python eval_retrieval.py --gold " + args.out)


if __name__ == "__main__":
    main()
