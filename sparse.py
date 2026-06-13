"""稀疏向量（BM25）封装，混合检索的"词面"那一路。

用 FastEmbed 的本地 BM25 模型生成稀疏向量（term -> 权重）。
- 文档侧 embed()：带文档长度归一化。
- 查询侧 query_embed()：BM25 查询不做长度归一化，IDF 由 Qdrant 的 IDF modifier 在库侧施加。
返回 (indices, values)，可直接喂给 Qdrant 的 SparseVector。
"""
from fastembed import SparseTextEmbedding

from config import SPARSE_MODEL

_model: SparseTextEmbedding | None = None


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _model


def embed_docs(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """文档侧稀疏向量，批量。"""
    return [
        (emb.indices.tolist(), emb.values.tolist())
        for emb in _get_model().embed(texts)
    ]


def embed_query(text: str) -> tuple[list[int], list[float]]:
    """查询侧稀疏向量。"""
    emb = next(iter(_get_model().query_embed(text)))
    return emb.indices.tolist(), emb.values.tolist()
