"""稀疏向量（BM25）封装，混合检索的"词面"那一路。

用 FastEmbed 的本地 BM25 模型生成稀疏向量（term -> 权重）。
- 文档侧 embed()：带文档长度归一化。
- 查询侧 query_embed()：BM25 查询不做长度归一化，IDF 由 Qdrant 的 IDF modifier 在库侧施加。
返回 (indices, values)，可直接喂给 Qdrant 的 SparseVector。

⚠️ 中文分词：FastEmbed 的 BM25 分词器只按空白/标点切词、**不分连续中文**，整段无标点
的中文会被压成一个 token——中文查询因此几乎匹配不上任何文档，sparse 一路形同虚设
（混合检索退化成纯 dense）。所以这里先用 jieba 把中文切成词、空格拼接再喂给 BM25，
**文档侧与查询侧必须用同一套切法**（`_segment`）以保证 term 对齐。
"""
import jieba

from fastembed import SparseTextEmbedding

from config import FASTEMBED_CACHE_DIR, SPARSE_MODEL


def _segment(text: str) -> str:
    """用 jieba 把连续中文切成词、以空格分隔，让下游 BM25 能按中文词匹配。

    用搜索引擎模式（cut_for_search）：对长词额外切出子词，提升召回。英文/数字 jieba
    原样保留，标点交给 BM25 自身的分词器处理。文档侧与查询侧共用本函数，确保一致。
    """
    return " ".join(w for w in jieba.cut_for_search(text) if w.strip())

_model: SparseTextEmbedding | None = None
def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        # 显式指定 cache_dir 指向项目内预置缓存，避免依赖 fastembed 的系统默认目录
        # （各平台不一致，且离线环境无法联网下载），clone 下来即开箱即用。
        _model = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=FASTEMBED_CACHE_DIR)
    return _model


def embed_docs(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """文档侧稀疏向量，批量。先 jieba 切词再交给 BM25。"""
    return [
        (emb.indices.tolist(), emb.values.tolist())
        for emb in _get_model().embed([_segment(t) for t in texts])
    ]


def embed_query(text: str) -> tuple[list[int], list[float]]:
    """查询侧稀疏向量。先 jieba 切词再交给 BM25（与文档侧同一套切法）。"""
    emb = next(iter(_get_model().query_embed(_segment(text))))
    return emb.indices.tolist(), emb.values.tolist()
