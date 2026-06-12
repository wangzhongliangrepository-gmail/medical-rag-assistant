"""统计 classify 节点对 bridge/comparison 题的误判率。

用法：
  python debug_classify.py        # 前 200 条
  python debug_classify.py -n 50  # 自定义条数
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm

from config import LLM_FLASH
from data import load_hotpotqa
from llm import get_llm
from nodes import classify


def check_one(llm, example) -> tuple[str, str]:
    """返回 (gold_type, pred_type)"""
    result = classify(
        {"question": example["question"]},
        llm=llm,
    )
    return example["type"], result["question_type"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    dataset = load_hotpotqa(n=args.n)
    llm = get_llm(model=LLM_FLASH)
    fn = partial(check_one, llm)

    # gold_type -> {pred_type -> count}
    confusion: dict[str, dict[str, int]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn, ex): ex for ex in dataset}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="classifying"):
            try:
                gold, pred = fut.result(timeout=60)
            except Exception as e:
                print(f"\n[WARN] {type(e).__name__}: {e}")
                continue
            row = confusion.setdefault(gold, {})
            row[pred] = row.get(pred, 0) + 1

    print("\n=== Classify Confusion Matrix ===")
    total_correct = 0
    total_count = 0
    for gold in ("bridge", "comparison"):
        row = confusion.get(gold, {})
        n = sum(row.values())
        correct = row.get(gold, 0)
        wrong = n - correct
        total_correct += correct
        total_count += n
        print(f"  gold={gold:<12} n={n:>4}  correct={correct:>4}  wrong={wrong:>4}  acc={correct/n:.3f}")
        for pred, cnt in sorted(row.items()):
            marker = "" if pred == gold else "  <-- MISCLASSIFIED"
            print(f"    pred={pred:<12} {cnt:>4}{marker}")

    if total_count:
        print(f"\n  overall accuracy: {total_correct/total_count:.3f}  ({total_correct}/{total_count})")


if __name__ == "__main__":
    main()
