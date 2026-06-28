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
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse
import json
import re
import shutil
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
    QDRANT_PATH,
    QDRANT_URL,
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

# 结构感知切分：医学教材常用【小节】标题（如【不良反应与防治】【禁忌】【药物相互作用】）。
# 按这些标题切，让每个小节自成 chunk（避免把"禁忌列表"从中间截断）；超长小节再用递归兜底。
_SECTION_RE = re.compile(r"(【[^】]{1,20}】)")
_STRUCT_MAX = CHUNK_SIZE * 2  # 小节长度封顶；超过则对该小节再做递归切分


def _structure_split(text: str) -> list[str]:
    """按【小节】标题切分，标题与其正文绑在一块；无标题或超长则退回递归切分。"""
    parts = _SECTION_RE.split(text)  # [前言, 【标题1】, 正文1, 【标题2】, 正文2, ...]
    sections: list[str] = []
    if parts[0].strip():             # 第一个【之前的前言（若有）
        sections.append(parts[0])
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append(header + body)  # 标题 + 其下正文，作为一个语义单元

    chunks: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= _STRUCT_MAX:
            chunks.append(sec)          # 小节整体成块
        else:
            chunks.extend(_SPLITTER.split_text(sec))  # 超长小节再切
    return chunks or _SPLITTER.split_text(text)       # 兜底：完全无结构时退回递归


def load_and_chunk(limit: int | None = None, strategy: str = "recursive") -> list[dict]:
    split = _structure_split if strategy == "structure" else _SPLITTER.split_text
    chunks: list[dict] = []
    with open(SOURCE, encoding="utf-8") as f:
        for source_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            text = json.loads(line)["text"]
            for ci, piece in enumerate(split(text)):
                chunks.append({"text": piece, "source_id": source_id, "chunk_id": ci})
            if limit and source_id + 1 >= limit:
                break
    return chunks


def _purge_collection_dir() -> None:
    """本地落盘模式下，只清【目标集合自己的目录】，绝不动同路径下别的集合。

    local Qdrant 把每个集合放在 <QDRANT_PATH>/collection/<集合名>/。
    （历史上这里曾 rmtree 整个 QDRANT_PATH，会误删同路径其它库——已废弃。）
    """
    if QDRANT_URL:               # 服务器模式：由 delete_collection 经 API 处理，无本地目录
        return
    coll_dir = Path(QDRANT_PATH) / "collection" / QDRANT_COLLECTION
    shutil.rmtree(coll_dir, ignore_errors=True)


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    exists = client.collection_exists(QDRANT_COLLECTION)
    if exists and recreate:
        client.delete_collection(QDRANT_COLLECTION)
        _purge_collection_dir()  # 兜底清残留，仅限该集合目录
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
    parser.add_argument("--recreate", action="store_true",
                        help="清空并重建【当前集合】（只动 QDRANT_COLLECTION 这一个，不碰同路径别的库）")
    parser.add_argument("--strategy", choices=["recursive", "structure"], default="recursive",
                        help="切分策略：recursive=递归字符切分（默认）；structure=按【小节】结构切分")
    args = parser.parse_args()

    print(f"[1/4] 读取并切块（strategy={args.strategy}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）...")
    chunks = load_and_chunk(args.limit, strategy=args.strategy)
    print(f"      源段落 {'(前%d条)' % args.limit if args.limit else '(全量)'} → {len(chunks)} 个 chunk")

    print("[2/4] 连接 Qdrant 并准备集合（dense + sparse）...")
    # 注意：--recreate 只重建【当前 QDRANT_COLLECTION 这一个集合】，不会动同路径下别的库。
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
