"""查看 planner 对 bridge 题的拆解结果。"""
from config import LLM_FLASH
from data import load_hotpotqa
from llm import get_llm
from nodes import planner

llm = get_llm(model=LLM_FLASH)
ds = load_hotpotqa(n=50)

count = 0
for ex in ds:
    if ex["type"] != "bridge":
        continue
    contexts = [
        {"title": t, "sentences": s}
        for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
    ]
    result = planner({"question": ex["question"], "raw_contexts": contexts, "sub_questions": [], "evidence": [], "top_contexts": [], "answer": ""}, llm=llm)
    print(f"Q:    {ex['question']}")
    print(f"SubQ: {result['sub_questions']}")
    print()
    count += 1
    if count >= 5:
        break
