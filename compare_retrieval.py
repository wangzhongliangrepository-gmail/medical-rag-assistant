"""混合检索价值演示：同一 query 跑 dense-only / sparse-only / hybrid，对比召回差异。

直观展示 dense（语义）与 sparse（词面）召回的是不同的集合，hybrid（RRF）取并集，
从而召回更全——这是"为什么要混合检索"的实证。

用法：
  python compare_retrieval.py "氨氯地平的降压机制"
  python compare_retrieval.py "吃药为什么要按体重调整剂量" --k 5
"""
import argparse

import sparse
from config import DENSE_VECTOR_NAME, QDRANT_COLLECTION, SPARSE_VECTOR_NAME
from embeddings import get_embeddings
from qdrant_client import models
from vectordb import get_client


def _ids_and_snippets(points) -> list[tuple[str, str]]:
    out = []
    for p in points:
        key = f"{p.payload['source_id']}-{p.payload['chunk_id']}"
        out.append((key, p.payload["text"][:60].replace("\n", " ")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5, help="每路取前 k 条")
    args = parser.parse_args()
    q, k = args.query, args.k

    client = get_client()
    dense_q = get_embeddings().embed_query(q)
    sidx, sval = sparse.embed_query(q)
    sparse_q = models.SparseVector(indices=sidx, values=sval)

    dense_hits = client.query_points(
        QDRANT_COLLECTION, query=dense_q, using=DENSE_VECTOR_NAME, limit=k, with_payload=True
    ).points
    sparse_hits = client.query_points(
        QDRANT_COLLECTION, query=sparse_q, using=SPARSE_VECTOR_NAME, limit=k, with_payload=True
    ).points
    hybrid_hits = client.query_points(
        QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_q, using=DENSE_VECTOR_NAME, limit=20),
            models.Prefetch(query=sparse_q, using=SPARSE_VECTOR_NAME, limit=20),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
        with_payload=True,
    ).points

    dense = _ids_and_snippets(dense_hits)
    sparse_ = _ids_and_snippets(sparse_hits)
    hybrid = _ids_and_snippets(hybrid_hits)

    dense_ids = {x[0] for x in dense}
    sparse_ids = {x[0] for x in sparse_}

    def show(title, items):
        print(f"\n--- {title} (top {k}) ---")
        for i, (key, snip) in enumerate(items, 1):
            tag = ""
            if title.startswith("HYBRID"):
                src = []
                if key in dense_ids:
                    src.append("D")
                if key in sparse_ids:
                    src.append("S")
                tag = f"  [来源:{'+'.join(src) or '?'}]"
            print(f"  {i}. ({key}) {snip}...{tag}")

    print(f"\nQ: {q}")
    show("DENSE-only", dense)
    show("SPARSE-only", sparse_)
    show("HYBRID(RRF)", hybrid)

    # 互补性统计
    only_dense = dense_ids - sparse_ids
    only_sparse = sparse_ids - dense_ids
    overlap = dense_ids & sparse_ids
    print(f"\n=== 互补性 ===")
    print(f"  dense 与 sparse 重叠: {len(overlap)}/{k}")
    print(f"  仅 dense 命中: {len(only_dense)}  |  仅 sparse 命中: {len(only_sparse)}")
    print(f"  → 重叠越少，说明两路召回越互补，hybrid 融合收益越大")


if __name__ == "__main__":
    main()
