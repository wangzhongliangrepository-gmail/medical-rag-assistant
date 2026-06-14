"""清理 Qdrant 集合里的重复 chunk（按 source_id+chunk_id 去重）。

本地 Qdrant 的 --recreate 未彻底 purge 旧落盘数据，导致小批测试数据与全量数据重复。
本脚本扫描全集，每个 (source_id, chunk_id) 只保留一个 point，删除多余的。
"""
from qdrant_client import models

from config import QDRANT_COLLECTION
from vectordb import get_client


def main() -> None:
    client = get_client()
    before = client.count(QDRANT_COLLECTION).count
    print(f"去重前：{before} 个 point")

    seen: set[tuple] = set()
    to_delete: list[str] = []
    offset = None
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=2000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            key = (p.payload["source_id"], p.payload["chunk_id"])
            if key in seen:
                to_delete.append(p.id)
            else:
                seen.add(key)
        scanned += len(points)
        if offset is None:
            break

    print(f"扫描 {scanned} 个 point，发现重复 {len(to_delete)} 个")
    if to_delete:
        # 分批删除
        for i in range(0, len(to_delete), 1000):
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.PointIdsList(points=to_delete[i : i + 1000]),
            )
    after = client.count(QDRANT_COLLECTION).count
    print(f"去重后：{after} 个 point（应 = 唯一 chunk 数 {len(seen)}）")


if __name__ == "__main__":
    main()
