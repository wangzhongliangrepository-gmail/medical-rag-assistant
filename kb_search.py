"""在 Qdrant 医学知识库上做检索（向量召回 → BGE 重排），验证 ingestion 成果。

用法：
  python kb_search.py "临床药理学研究什么"
  python kb_search.py "高血压怎么治" --topk 5
"""
import argparse

from config import QDRANT_COLLECTION, RERANK_TOP_K
from embeddings import get_embeddings, rerank
from ingest import get_client


def search(query: str, recall: int = 20, top_k: int = RERANK_TOP_K) -> list[dict]:
    """两段式检索：向量召回 recall 条 → BGE 重排取 top_k。"""
    client = get_client()
    qvec = get_embeddings().embed_query(query)
    hits = client.query_points(
        collection_name=QDRANT_COLLECTION, query=qvec, limit=recall
    ).points
    if not hits:
        return []
    docs = [h.payload["text"] for h in hits]
    ranked = rerank(query, docs, top_n=top_k)
    return [
        {"score": r["relevance_score"], "text": docs[r["index"]]}
        for r in ranked
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--recall", type=int, default=20, help="向量召回条数")
    parser.add_argument("--topk", type=int, default=RERANK_TOP_K, help="重排后保留条数")
    args = parser.parse_args()

    results = search(args.query, recall=args.recall, top_k=args.topk)
    if not results:
        print("（无结果：知识库可能为空，先跑 python ingest.py）")
        return
    print(f"\nQ: {args.query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.3f}")
        print(f"    {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
