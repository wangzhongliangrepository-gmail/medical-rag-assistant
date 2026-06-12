"""M1 基线评测：在 HotpotQA distractor validation 上跑 EM / F1。

用法：
  python eval.py          # 快速验证（200 条）
  python eval.py --full   # 全量（7405 条）
  python eval.py -n 50    # 自定义条数
"""
import argparse
import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from data import load_hotpotqa
from graph import get_graph


# ---------- 答案归一化（HotpotQA 官方口径）----------

def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


# ---------- HotpotQA context 转 raw_contexts ----------

def to_raw_contexts(example) -> list[dict]:
    titles = example["context"]["title"]
    sentences_list = example["context"]["sentences"]
    return [{"title": t, "sentences": s} for t, s in zip(titles, sentences_list)]


# ---------- 单条评测 ----------

def eval_one(graph, example) -> tuple[float, float, str, int]:
    result = graph.invoke({
        "question": example["question"],
        "raw_contexts": to_raw_contexts(example),
    })
    pred = result["answer"]
    gold = example["answer"]
    revisions = result.get("revisions", 0)
    return exact_match(pred, gold), f1(pred, gold), example["type"], revisions


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="跑全量 7405 条")
    parser.add_argument("-n", type=int, default=200, help="自定义条数（--full 时忽略）")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--no-reflect", action="store_true", help="ablation：关掉反思回路")
    args = parser.parse_args()

    n = None if args.full else args.n
    dataset = load_hotpotqa(n=n)
    graph = get_graph(use_reflect=not args.no_reflect)

    em_total, f1_total, rev_total = 0.0, 0.0, 0
    buckets: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(eval_one, graph, ex): ex for ex in dataset}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluating"):
            try:
                em, f1_score, qtype, revisions = fut.result(timeout=180)
            except Exception as e:
                print(f"\n[WARN] sample failed: {type(e).__name__}: {e}")
                continue
            em_total += em
            f1_total += f1_score
            rev_total += revisions
            b = buckets.setdefault(qtype, {"em": 0.0, "f1": 0.0, "n": 0})
            b["em"] += em
            b["f1"] += f1_score
            b["n"] += 1

    count = len(dataset)
    mode = "no-reflect" if args.no_reflect else "reflect"
    print(f"\n=== Results (n={count}, top-3, {mode}) ===")
    print(f"{'total':<12} EM={em_total/count:.4f}  F1={f1_total/count:.4f}  (n={count})")
    for qtype in ("bridge", "comparison"):
        if qtype in buckets:
            b = buckets[qtype]
            print(f"{qtype:<12} EM={b['em']/b['n']:.4f}  F1={b['f1']/b['n']:.4f}  (n={b['n']})")
    print(f"avg_revisions: {rev_total/count:.2f}")


if __name__ == "__main__":
    main()