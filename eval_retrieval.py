"""检索评测：对一组「问题 → 金标 chunk」，对比三种检索方式的 Recall@k / Hit@k / MRR。

三种方式（逐层加码，看每一层带来的增益）：
  ① dense          —— 纯稠密语义检索（只 bge-m3 向量，无 sparse、无重排）
  ② hybrid (RRF)   —— dense + sparse 双路 RRF 融合（= kb_search.hybrid_recall）
  ③ hybrid+rerank  —— 融合后再过 BGE reranker 精排（= 线上 kb_search.search 的完整路径）

金标 chunk 用 (source_id, chunk_id) 唯一标识，和库里 payload 对齐。

指标（对每个 k，跨所有问题求平均）：
  - Hit@k    : 前 k 条里**至少**命中一个金标的问题占比（够不够"沾上"）
  - Recall@k : 前 k 条里命中的金标数 / 该题金标总数（够不够"全"）
  - MRR@k    : 第一个金标命中的排名倒数（最相关的排得够不够前）

金标文件格式（gen_eval_set.py 产出、人工审核后）：
  [{"question": "...", "gold": [[source_id, chunk_id], ...]}, ...]

用法：
  python eval_retrieval.py --gold eval_set_draft.json
  python eval_retrieval.py --gold eval_set.json --k 1 3 5 10
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse
import json

from config import (
    DENSE_VECTOR_NAME,
    QDRANT_COLLECTION,
    RECALL_K,
    SPARSE_VECTOR_NAME,
)

import sparse
from embeddings import get_embeddings, rerank
from qdrant_client import models
from vectordb import get_client

# ---- 单例：本地落盘模式下多开 QdrantClient 会抢文件锁，全程共用一个 ----
_CLIENT = None
_EMB = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = get_client()
    return _CLIENT


def _emb():
    global _EMB
    if _EMB is None:
        _EMB = get_embeddings()
    return _EMB


def _payload_to_cand(p) -> dict:
    return {"source_id": p["source_id"], "chunk_id": p["chunk_id"], "text": p["text"]}


# ---------- 三种检索方式（都返回有序候选 [{source_id, chunk_id, text}]）----------

def dense_recall(query: str, recall: int = RECALL_K) -> list[dict]:
    """① 纯 dense：只用 bge-m3 稠密向量。"""
    hits = _client().query_points(
        collection_name=QDRANT_COLLECTION,
        query=_emb().embed_query(query),
        using=DENSE_VECTOR_NAME,
        limit=recall,
        with_payload=True,
    ).points
    return [_payload_to_cand(h.payload) for h in hits]


def hybrid_recall(query: str, recall: int = RECALL_K) -> list[dict]:
    """② 混合：dense + sparse 双路 RRF 融合（复刻 kb_search，用共享 client）。"""
    sidx, sval = sparse.embed_query(query)
    hits = _client().query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=_emb().embed_query(query), using=DENSE_VECTOR_NAME, limit=recall),
            models.Prefetch(
                query=models.SparseVector(indices=sidx, values=sval),
                using=SPARSE_VECTOR_NAME, limit=recall,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=recall,
        with_payload=True,
    ).points
    return [_payload_to_cand(h.payload) for h in hits]


def hybrid_rerank(query: str, recall: int = RECALL_K) -> list[dict]:
    """③ 混合 + 重排：对融合候选整体重排（top_n=全部，得到完整新顺序）。"""
    cands = hybrid_recall(query, recall)
    if not cands:
        return []
    ranked = rerank(query, [c["text"] for c in cands], top_n=len(cands))
    return [cands[r["index"]] for r in ranked]


METHODS = {
    "dense": dense_recall,
    "hybrid(RRF)": hybrid_recall,
    "hybrid+rerank": hybrid_rerank,
}


# ---------- 指标 ----------

def _k(c: dict) -> tuple:
    return (c["source_id"], c["chunk_id"])


def metrics_at_k(ranked: list[dict], gold: set, k: int) -> tuple[float, float, float]:
    """返回 (hit@k, recall@k, mrr@k)。"""
    keys = [_k(c) for c in ranked[:k]]
    found = set(keys) & gold
    hit = 1.0 if found else 0.0
    recall = len(found) / len(gold)
    rr = 0.0
    for rank, key in enumerate(keys, 1):
        if key in gold:
            rr = 1.0 / rank
            break
    return hit, recall, rr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="金标 JSON：[{question, gold:[[sid,cid]]}]")
    parser.add_argument("--recall", type=int, default=RECALL_K, help="每路召回条数")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10], help="评测的 k 值")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        data = json.load(f)
    items = [d for d in data if d.get("question") and d.get("gold")]
    ks = [k for k in sorted(set(args.k)) if k <= args.recall]
    print(f"评测集 {len(items)} 题（已过滤无金标的）；recall={args.recall}；k={ks}\n")

    # 累加器：{method: {k: [hit_sum, recall_sum, mrr_sum]}}
    agg = {m: {k: [0.0, 0.0, 0.0] for k in ks} for m in METHODS}

    for i, item in enumerate(items, 1):
        q = item["question"]
        gold = {(sid, cid) for sid, cid in item["gold"]}
        for m, fn in METHODS.items():
            try:
                ranked = fn(q, recall=args.recall)
            except Exception as e:
                print(f"  [{i}] 方法 {m} 检索失败：{type(e).__name__}: {e}")
                continue
            for k in ks:
                hit, rec, rr = metrics_at_k(ranked, gold, k)
                agg[m][k][0] += hit
                agg[m][k][1] += rec
                agg[m][k][2] += rr
        if i % 10 == 0:
            print(f"  已评 {i}/{len(items)}...")

    n = len(items)
    print("\n" + "=" * 64)
    print(f"{'方法':<16}{'k':>4}{'Hit@k':>10}{'Recall@k':>12}{'MRR@k':>10}")
    print("-" * 64)
    for m in METHODS:
        for k in ks:
            h, r, rr = (s / n for s in agg[m][k])
            print(f"{m:<16}{k:>4}{h:>10.3f}{r:>12.3f}{rr:>10.3f}")
        print("-" * 64)
    print("解读：三种方式同一 k 横向比，越往下（加 sparse、加 rerank）指标应越高，"
          "即可量化「混合检索 / 重排」各自的增益。")


if __name__ == "__main__":
    main()
