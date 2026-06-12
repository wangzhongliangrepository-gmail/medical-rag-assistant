"""验证 M3 reflect 节点是否正常触发和终止。"""
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
            print(f"[planner]       sub_questions={updates['sub_questions']}")
        elif node == "refine" and updates:
            print(f"[refine]        精化后={updates['sub_questions'][0]!r}")
        elif node == "answer":
            print(f"[answer]        {updates['answer']!r}")
        elif node == "reflect":
            print(f"[reflect]       revisions={updates['revisions']}  sufficient={not bool(updates.get('missing_info'))}")
            print(f"                reasoning={updates['reflection']!r}")
            if updates.get("missing_info"):
                print(f"                missing={updates['missing_info']!r}")
        elif node == "prepare_retry":
            print(f"[prepare_retry] 新子问题={updates['sub_questions']}")

    count += 1
    if count >= 3:
        break