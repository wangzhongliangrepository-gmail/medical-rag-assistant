"""一键对比两种切分策略的检索效果（子集实验，source 级公平匹配）。

**自给自足**：只用「前 N 段落」建两个小子集做对比，不依赖、不触碰你的线上全量库
medical_kb。两次灌库各用**独立子进程 + 独立 Qdrant 路径**，彼此和线上库完全隔离
（ingest --recreate 在本地模式会 rmtree 整个路径，独立路径才不会误删别的库）。

  ① baseline 切分（默认 recursive）灌前 N 段落 → qdrant_db_base / medical_kb_base
  ② 新切分（默认 structure）灌前 N 段落    → qdrant_db_v2   / medical_kb_v2
  ③/④ 两边各评（source 级匹配，同一批金标段落<N 的题）
  ⑤ 并排打印「旧 vs 新」Recall@k / MRR@k

用法：
  python run_chunk_eval.py                                       # N=3000，recursive vs structure
  python run_chunk_eval.py --limit 4000
  python run_chunk_eval.py --skip-ingest                         # 两子集已灌过，只跑对比

前置：Xinference 开着；这两个独立路径没有别的进程占用。
（注：本实验不需要、也不会重建全量 medical_kb；想恢复线上库另跑 `python ingest.py --recreate`。）
"""
import argparse
import json
import os
import subprocess
import sys

NEW_PATH = "./qdrant_db_v2"        # 新切分独立落盘路径
NEW_COLL = "medical_kb_v2"
BASE_PATH = "./qdrant_db_base"     # 旧切分独立落盘路径（不是线上 qdrant_db！）
BASE_COLL = "medical_kb_base"
RESULT_PREFIX = "__RESULTS_JSON__"


def _run(cmd: list[str], env_overrides: dict, capture: bool) -> str:
    """跑子进程；env_overrides 覆盖 QDRANT_PATH/QDRANT_COLLECTION 等做隔离。"""
    env = dict(os.environ, **env_overrides)
    print(f"  $ {' '.join(cmd)}   [{env_overrides}]")
    if capture:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
        if p.returncode != 0:
            print(p.stdout); print(p.stderr)
            raise SystemExit(f"子进程失败：{cmd}")
        return p.stdout
    subprocess.run(cmd, env=env, check=True)
    return ""


def _parse_results(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    raise SystemExit("未在子进程输出里找到结果 JSON（eval_retrieval 是否带了 --json？）")


def _eval(path: str, coll: str, gold: str, limit: int, recall: int, ks: list[int]) -> dict:
    out = _run(
        [sys.executable, "eval_retrieval.py", "--gold", gold,
         "--match", "source", "--max-source", str(limit),
         "--recall", str(recall), "--k", *map(str, ks), "--json"],
        {"QDRANT_PATH": path, "QDRANT_COLLECTION": coll}, capture=True)
    return _parse_results(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="eval_set_draft.json")
    p.add_argument("--limit", type=int, default=3000, help="子集：前 N 段落")
    p.add_argument("--recall", type=int, default=20)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--baseline-strategy", default="recursive", choices=["recursive", "structure"],
                   help="baseline 切分策略（默认 recursive）")
    p.add_argument("--strategy", default="structure", choices=["recursive", "structure"],
                   help="新切分策略（默认 structure）")
    p.add_argument("--skip-ingest", action="store_true", help="两子集已灌好，跳过①②直接对比")
    args = p.parse_args()
    ks = sorted(set(args.k))

    # ①② 各建一个 N 段落子集到独立路径（绝不碰线上 medical_kb）
    if not args.skip_ingest:
        print(f"[①] baseline 切分 {args.baseline_strategy} 灌前 {args.limit} 段落 → {BASE_PATH}/{BASE_COLL}")
        _run([sys.executable, "ingest.py", "--recreate",
              "--limit", str(args.limit), "--strategy", args.baseline_strategy],
             {"QDRANT_PATH": BASE_PATH, "QDRANT_COLLECTION": BASE_COLL}, capture=False)
        print(f"\n[②] 新切分 {args.strategy} 灌前 {args.limit} 段落 → {NEW_PATH}/{NEW_COLL}")
        _run([sys.executable, "ingest.py", "--recreate",
              "--limit", str(args.limit), "--strategy", args.strategy],
             {"QDRANT_PATH": NEW_PATH, "QDRANT_COLLECTION": NEW_COLL}, capture=False)

    # ③ 旧切分
    print(f"\n[③] 评 baseline（{BASE_PATH}/{BASE_COLL}）")
    old = _eval(BASE_PATH, BASE_COLL, args.gold, args.limit, args.recall, ks)

    # ④ 新切分
    print(f"\n[④] 评 新切分（{NEW_PATH}/{NEW_COLL}）")
    new = _eval(NEW_PATH, NEW_COLL, args.gold, args.limit, args.recall, ks)

    if old["n"] != new["n"]:
        print(f"⚠️ 两侧题数不一致（旧 {old['n']} / 新 {new['n']}），对比可能不公平。")

    # ④ 并排
    print("\n" + "=" * 80)
    print(f"切分对比（子集前 {args.limit} 段落 · {old['n']} 题 · source 级匹配）"
          f"  旧={args.baseline_strategy}  新={args.strategy}")
    print("=" * 80)
    print(f"{'方法':<15}{'k':>3} | {'旧 R@k':>8}{'新 R@k':>8}{'ΔR':>7} | "
          f"{'旧 MRR':>8}{'新 MRR':>8}{'ΔMRR':>8}")
    print("-" * 80)
    for m in old["results"]:
        for k in ks:
            o = old["results"][m][str(k)]
            nw = new["results"][m][str(k)]
            dR, dM = nw["recall"] - o["recall"], nw["mrr"] - o["mrr"]
            print(f"{m:<15}{k:>3} | {o['recall']:>8.3f}{nw['recall']:>8.3f}{dR:>+7.3f} | "
                  f"{o['mrr']:>8.3f}{nw['mrr']:>8.3f}{dM:>+8.3f}")
        print("-" * 80)
    print("ΔR=新-旧 Recall@k（source 级）；正=结构切分更好。重点看 hybrid+rerank（线上完整路径）。")
    print("注：source 级 recall 照不出『不截断小节→答案更完整』的收益，那需 answer 级 LLM-judge。")


if __name__ == "__main__":
    main()