"""在 Qdrant 医疗库上做混合检索：dense + sparse → RRF 融合 → BGE 重排。

两段式：
  ① 召回：dense(bge-m3) 与 sparse(BM25) 各召回 RECALL_K 条 → Qdrant RRF 融合
  ② 精排：bge-reranker-v2-m3 对融合候选重排，取 top_k

用法：
  python kb_search.py "氨氯地平能降血压吗"
  python kb_search.py "高血压怎么治" --topk 5
"""
import argparse

import sparse
from config import (
    DENSE_VECTOR_NAME,
    QDRANT_COLLECTION,
    RECALL_K,
    RERANK_TOP_K,
    SPARSE_VECTOR_NAME,
)
from embeddings import get_embeddings, rerank
from qdrant_client import models
from vectordb import get_client


def hybrid_recall(query: str, recall: int = RECALL_K) -> list[str]:
    """dense + sparse 双路召回，Qdrant 内 RRF 融合，返回候选原文。"""
    client = get_client()
    dense_q = get_embeddings().embed_query(query)
    sidx, sval = sparse.embed_query(query)

    hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_q, using=DENSE_VECTOR_NAME, limit=recall),
            models.Prefetch(
                query=models.SparseVector(indices=sidx, values=sval),
                using=SPARSE_VECTOR_NAME,
                limit=recall,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=recall,
        with_payload=True,
    ).points
    return [h.payload["text"] for h in hits]


def search(query: str, recall: int = RECALL_K, top_k: int = RERANK_TOP_K) -> list[dict]:
    docs = hybrid_recall(query, recall)
    if not docs:
        return []
    ranked = rerank(query, docs, top_n=top_k)
    return [{"score": r["relevance_score"], "text": docs[r["index"]]} for r in ranked]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--recall", type=int, default=RECALL_K, help="每路召回条数")
    parser.add_argument("--topk", type=int, default=RERANK_TOP_K, help="重排后保留条数")
    args = parser.parse_args()

    results = search(args.query, recall=args.recall, top_k=args.topk)
    if not results:
        print("（无结果：知识库可能为空，先跑 python ingest.py --recreate）")
        return
    print(f"\nQ: {args.query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.3f}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
