from config import RERANK_TOP_K
from data import load_hotpotqa
from embeddings import rerank
from graph import get_graph

ex = load_hotpotqa(n=1)[0]

question = ex["question"]
gold_answer = ex["answer"]
gold_titles = set(ex["supporting_facts"]["title"])
titles = ex["context"]["title"]
sentences = ex["context"]["sentences"]

docs = [t + " " + " ".join(s) for t, s in zip(titles, sentences)]
results = rerank(question, docs, top_n=RERANK_TOP_K)

print("=" * 60)
print("问题:", question)
print()
print(f"Top-{RERANK_TOP_K} 检索结果：")
hit = 0
for i, r in enumerate(results):
    title = titles[r["index"]]
    is_gold = "[gold]" if title in gold_titles else "[dist]"
    if title in gold_titles:
        hit += 1
    print(f"  [{i+1}] {is_gold}  {title}  (score={r['relevance_score']:.4f})")

print()
print(f"金标段落共 {len(gold_titles)} 篇，top-{RERANK_TOP_K} 命中 {hit} 篇")
print()

g = get_graph("flash")
contexts = [{"title": t, "sentences": s} for t, s in zip(titles, sentences)]
result = g.invoke({"question": question, "raw_contexts": contexts})

print("模型答案:", result["answer"])
print("金标答案:", gold_answer)
print("=" * 60)