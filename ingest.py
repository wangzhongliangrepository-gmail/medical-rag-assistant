"""把医学教材语料灌进 Qdrant 持久向量库（P0：内部知识库 ingestion）。

流程：读 medical_book_zh.json（JSONL）→ 重切 ~400 字 → BGE 嵌入 → 入 Qdrant。

为什么要重切：medical_book 原始段落按 2048 字切，远超 BGE-large-zh 的 512 token
上限，直接嵌入会被截断、丢信息。这里用 RecursiveCharacterTextSplitter 按中文标点
优先切到 CHUNK_SIZE 以内，相邻块留 CHUNK_OVERLAP 重叠防止切断语义。

用法：
  python ingest.py --limit 200    # 先小批验证（200 条教材段）
  python ingest.py                # 全量 8475 条
  python ingest.py --recreate     # 清空集合重建
"""
import argparse
import json
import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_DIM,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
)
from embeddings import get_embeddings

SOURCE = Path(__file__).parent / "medical_data" / "pretrain" / "medical_book_zh.json"
EMBED_BATCH = 64  # 每批嵌入的 chunk 数，避免单次请求过大

# 中文优先按段落/句子切，最后才退化到字符
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


def get_client() -> QdrantClient:
    """开发期连本地落盘 Qdrant；设了 QDRANT_URL 则连服务器（部署期）。"""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=QDRANT_PATH)


def load_and_chunk(limit: int | None = None) -> list[dict]:
    """读教材 JSONL → 重切 → 返回 [{text, source_id, chunk_id}, ...]。"""
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
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条教材段（验证用）")
    parser.add_argument("--recreate", action="store_true", help="清空集合重建")
    args = parser.parse_args()

    print(f"[1/4] 读取并切块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    chunks = load_and_chunk(args.limit)
    print(f"      源段落 {'(前%d条)' % args.limit if args.limit else '(全量)'} → {len(chunks)} 个 chunk")

    print("[2/4] 连接 Qdrant 并准备集合...")
    client = get_client()
    ensure_collection(client, args.recreate)

    print("[3/4] BGE 嵌入并写入...")
    embedder = get_embeddings()
    written = 0
    for i in tqdm(range(0, len(chunks), EMBED_BATCH), desc="ingesting"):
        batch = chunks[i : i + EMBED_BATCH]
        vectors = embedder.embed_documents([c["text"] for c in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": c["text"],
                    "source_id": c["source_id"],
                    "chunk_id": c["chunk_id"],
                    "source": "medical_book_zh",
                },
            )
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        written += len(points)

    count = client.count(QDRANT_COLLECTION).count
    print(f"[4/4] 完成：本次写入 {written}，集合 '{QDRANT_COLLECTION}' 现有 {count} 个向量。")


if __name__ == "__main__":
    main()
