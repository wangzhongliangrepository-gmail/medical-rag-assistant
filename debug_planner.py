"""逐节点打印，验证 refine 是否真的精化了子问题。"""
from data import load_hotpotqa
from graph import get_graph

ds = load_hotpotqa(n=50)
g = get_graph("flash")

count = 0
for ex in ds:
    if ex["type"] != "bridge":
        continue

    contexts = [
        {"title": t, "sentences": s}
        for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
    ]

    print("=" * 60)
    print(f"Q: {ex['question']}")
    print(f"Gold: {ex['answer']}")
    print()

    for step in g.stream({"question": ex["question"], "raw_contexts": contexts}, stream_mode="updates"):
        node, updates = next(iter(step.items()))
        if node == "planner":
            print(f"[planner] 子问题: {updates['sub_questions']}")
        elif node == "retrieve":
            print(f"[retrieve] 处理子问题，剩余: {updates['sub_questions']}")
        elif node == "refine":
            if updates:
                print(f"[refine]  中间答案: {updates.get('intermediate_answers', [])[-1]!r}")
                print(f"[refine]  精化后子问题: {updates['sub_questions'][0]!r}")
        elif node == "answer":
            print(f"[answer]  {updates['answer']!r}")

    count += 1
    if count >= 3:
        break
