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
    args = parser.parse_args()

    graph = get_med_graph()
    result = graph.invoke({"question": args.question})

    print(f"\nQ: {args.question}\n")
    print(f"A: {result['answer']}\n")
    print("来源:")
    for i, e in enumerate(result["evidence"], 1):
        snippet = e["text"][:50].replace("\n", " ")
        print(f"  [{i}] 教材段#{e['source_id']} (score={e['score']:.3f}) {snippet}...")
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
