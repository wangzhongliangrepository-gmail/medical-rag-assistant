"""在 Qdrant 医疗库上做混合检索：dense + sparse → RRF 融合 → BGE 重排。

返回带来源元数据（source_id / chunk_id）的证据，供带引用作答使用。

用法（直接看检索结果）：
  python kb_search.py "氨氯地平能降血压吗"
  python kb_search.py "高血压怎么治" --topk 5
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
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


def hybrid_recall(query: str, recall: int = RECALL_K) -> list[dict]:
    """dense + sparse 双路召回 + Qdrant RRF 融合，返回带元数据的候选。

    每项：{"text", "source_id", "chunk_id"}
    """
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
    return [
        {
            "text": h.payload["text"],
            "source_id": h.payload["source_id"],
            "chunk_id": h.payload["chunk_id"],
        }
        for h in hits
    ]


def search(query: str, recall: int = RECALL_K, top_k: int = RERANK_TOP_K) -> list[dict]:
    """混合召回 + BGE 重排，返回 top_k 带 score 与来源的证据。

    每项：{"text", "source_id", "chunk_id", "score"}
    """
    cands = hybrid_recall(query, recall)
    if not cands:
        return []
    docs = [c["text"] for c in cands]
    ranked = rerank(query, docs, top_n=top_k)
    return [{**cands[r["index"]], "score": r["relevance_score"]} for r in ranked]


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
        print(f"[{i}] score={r['score']:.3f}  教材段#{r['source_id']}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
