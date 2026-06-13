"""把医学教材语料灌进 Qdrant 混合检索库（P0/Phase2：dense + sparse）。

流程：读 medical_book_zh.json（JSONL）→ 重切 → 同时算 dense(bge-m3) 与 sparse(BM25)
     → 写入 Qdrant 命名向量集合（dense + sparse）。

每个 chunk 存两个向量：
- dense ：bge-m3 语义向量（1024 维，COSINE）
- sparse：BM25 词面稀疏向量（IDF 由 Qdrant 在库侧施加）
检索时两路并行召回 → RRF 融合 → 重排（见 kb_search.py）。

用法：
  python ingest.py --limit 200 --recreate   # 小批验证
  python ingest.py --recreate               # 全量 8475 条
"""
import argparse
import json
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from tqdm import tqdm

import sparse
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_VECTOR_NAME,
    EMBED_DIM,
    QDRANT_COLLECTION,
    SPARSE_VECTOR_NAME,
)
from embeddings import get_embeddings
from vectordb import get_client

SOURCE = Path(__file__).parent / "medical_data" / "pretrain" / "medical_book_zh.json"
EMBED_BATCH = 64

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


def load_and_chunk(limit: int | None = None) -> list[dict]:
    chunks: list[dict] = []
    with open(SOURCE, encoding="utf-8") as f:
        for source_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            for ci, piece in enumerate(_SPLITTER.split_text(text)):
                chunks.append({"text": piece, "source_id": source_id, "chunk_id": ci})
            if limit and source_id + 1 >= limit:
                break
    return chunks


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(QDRANT_COLLECTION)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                # IDF modifier：BM25 的 IDF 部分由 Qdrant 在库侧按全集统计施加
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条教材段")
    parser.add_argument("--recreate", action="store_true", help="清空集合重建")
    args = parser.parse_args()

    print(f"[1/4] 读取并切块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    chunks = load_and_chunk(args.limit)
    print(f"      源段落 {'(前%d条)' % args.limit if args.limit else '(全量)'} → {len(chunks)} 个 chunk")

    print("[2/4] 连接 Qdrant 并准备集合（dense + sparse）...")
    client = get_client()
    ensure_collection(client, args.recreate)

    print("[3/4] 计算 dense(bge-m3) + sparse(BM25) 并写入...")
    embedder = get_embeddings()
    written = 0
    for i in tqdm(range(0, len(chunks), EMBED_BATCH), desc="ingesting"):
        batch = chunks[i : i + EMBED_BATCH]
        texts = [c["text"] for c in batch]
        dense_vecs = embedder.embed_documents(texts)
        sparse_vecs = sparse.embed_docs(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: dvec,
                    SPARSE_VECTOR_NAME: models.SparseVector(indices=sidx, values=sval),
                },
                payload={
                    "text": c["text"],
                    "source_id": c["source_id"],
                    "chunk_id": c["chunk_id"],
                    "source": "medical_book_zh",
                },
            )
            for c, dvec, (sidx, sval) in zip(batch, dense_vecs, sparse_vecs)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        written += len(points)

    count = client.count(QDRANT_COLLECTION).count
    print(f"[4/4] 完成：本次写入 {written}，集合 '{QDRANT_COLLECTION}' 现有 {count} 个向量。")


if __name__ == "__main__":
    main()
