from data import load_hotpotqa
from graph import get_graph

ex = load_hotpotqa(n=1)[0]
g = get_graph("flash")
contexts = [
    {"title": t, "sentences": s}
    for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
]
result = g.invoke({"question": ex["question"], "raw_contexts": contexts})
print("Q:   ", ex["question"])
print("A:   ", result["answer"])
print("Gold:", ex["answer"])
