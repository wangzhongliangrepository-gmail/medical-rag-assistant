"""只重算并更新库里每个 chunk 的 sparse(BM25) 向量，dense 与 payload 原封不动。

为什么单独有这个脚本：sparse 的中文分词修复（sparse.py 引入 jieba）后，**文档侧**已落库的
旧 sparse 向量是「未分词」的，跟新的查询侧 term 对不上，必须重算文档侧。但 dense(bge-m3)
没变、且要走 Xinference GPU 很贵，没必要重灌——这里用 Qdrant 的 update_vectors 只覆盖
命名向量 sparse，dense 和 payload 都不碰。IDF 由 Qdrant 库侧按全集统计，全部更新完即一致。

用法（对每个库各跑一次，靠环境变量指定库）：
  QDRANT_PATH=./qdrant_db_base QDRANT_COLLECTION=medical_kb_base python reindex_sparse.py
  QDRANT_PATH=./qdrant_db_v2   QDRANT_COLLECTION=medical_kb_v2   python reindex_sparse.py
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载

from qdrant_client import models
from tqdm import tqdm

import sparse
from config import QDRANT_COLLECTION, SPARSE_VECTOR_NAME
from vectordb import get_client

BATCH = 256


def main() -> None:
    client = get_client()
    total = client.count(QDRANT_COLLECTION).count
    print(f"重算 sparse：集合 '{QDRANT_COLLECTION}'，共 {total} 个点（dense/payload 不动）")

    updated = 0
    offset = None
    with tqdm(total=total, desc="reindex sparse") as bar:
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=BATCH,
                offset=offset,
                with_payload=["text"],   # 只取 text，省内存
                with_vectors=False,      # 不取向量，dense 我们根本不碰
            )
            if not points:
                break
            texts = [p.payload["text"] for p in points]
            sparse_vecs = sparse.embed_docs(texts)
            client.update_vectors(
                collection_name=QDRANT_COLLECTION,
                points=[
                    models.PointVectors(
                        id=p.id,
                        vector={SPARSE_VECTOR_NAME: models.SparseVector(indices=sidx, values=sval)},
                    )
                    for p, (sidx, sval) in zip(points, sparse_vecs)
                ],
            )
            updated += len(points)
            bar.update(len(points))
            if offset is None:           # 没有下一页了
                break

    print(f"完成：更新 {updated} 个点的 sparse 向量。")


if __name__ == "__main__":
    main()