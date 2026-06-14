"""医疗 RAG CLI：问一句 → 混合检索 → DeepSeek 带引用作答。

用法：
  python med_rag.py "高血压的一线治疗药物有哪些"
  python med_rag.py "二甲双胍有什么副作用"
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse

from med_graph import get_med_graph

DISCLAIMER = "⚠️ 本回答仅供学习演示，非医疗建议；如有健康问题请咨询专业医师。"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", help="医疗问题")
    parser.add_argument("--web", action="store_true",
                        help="联网搜索：融合外部 Web（默认关，仅查内部教材库，更快）")
    args = parser.parse_args()

    graph = get_med_graph()
    result = graph.invoke({"question": args.question, "use_external": args.web})

    mode = "内部教材 + 联网 Web" if args.web else "仅内部教材"
    print(f"\n[模式] {mode}")
    print(f"\nQ: {args.question}\n")
    print("规划子问题:")
    for sq in result.get("sub_questions", []):
        print(f"  - {sq}")
    print()
    print(f"A: {result['answer']}\n")
    n_int = len(result.get("internal_evidence", []))
    n_ext = len(result.get("external_evidence", []))
    print(f"融合来源（内部教材 {n_int} 段 + 外部 Web {n_ext} 条 → 重排取 {len(result['evidence'])}）:")
    for i, e in enumerate(result["evidence"], 1):
        snippet = e["text"][:50].replace("\n", " ")
        print(f"  [{i}] (score={e['score']:.3f}) {e['source'][:60]}")
        print(f"       {snippet}...")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
