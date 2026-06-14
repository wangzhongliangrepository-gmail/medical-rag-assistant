"""医疗 RAG CLI：问一句 → 混合检索 → DeepSeek 带引用作答。

单次：
  python med_rag.py "高血压的一线治疗药物有哪些"
  python med_rag.py "二甲双胍有什么副作用" --web

多轮（带记忆，可指代消解 + 长期用户记忆）：
  python med_rag.py --chat            # 进入交互循环，输入 exit/quit 退出
"""
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse
import uuid

from med_graph import get_med_graph

DISCLAIMER = "⚠️ 本回答仅供学习演示，非医疗建议；如有健康问题请咨询专业医师。"


def _print_result(question: str, result: dict, show_sources: bool = True) -> None:
    standalone = result.get("standalone_question", question)
    if standalone and standalone != question:
        print(f"  （理解为：{standalone}）")
    mem = result.get("user_memory") or []
    if mem:
        print(f"  （已纳入用户记忆：{'；'.join(mem)}）")
    rev = result.get("revisions", 0)
    if rev > 1:
        print(f"  （🔁 经过 {rev} 轮证据反思核验，补检索后重答；最近反思：{result.get('reflection', '')}）")
    print(f"\nA: {result['answer']}\n")
    if show_sources:
        n_int = len(result.get("internal_evidence", []))
        n_ext = len(result.get("external_evidence", []))
        print(f"融合来源（内部教材 {n_int} 段 + 外部 Web {n_ext} 条 → 重排取 {len(result.get('evidence', []))}）:")
        for i, e in enumerate(result.get("evidence", []), 1):
            snippet = e["text"][:50].replace("\n", " ")
            print(f"  [{i}] (score={e['score']:.3f}) {e['source'][:60]}")
            print(f"       {snippet}...")


def run_chat(graph, use_external: bool, model_tier: str = "flash") -> None:
    """多轮交互：固定 user_id + thread_id，连续对话，记忆生效。"""
    user_id = "cli-user"
    thread_id = f"cli-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id, "model_tier": model_tier}}
    print(f"进入多轮对话模式（输入 exit / quit 退出）。  [模型 {model_tier}]"
          + ("  [联网开]" if use_external else "  [仅内部]"))
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出"):
            print("再见。")
            break
        result = graph.invoke({"question": q, "use_external": use_external}, config=config)
        _print_result(q, result, show_sources=False)
    print(f"\n{DISCLAIMER}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="医疗问题（不带则需 --chat）")
    parser.add_argument("--web", action="store_true",
                        help="联网搜索：融合外部 Web（默认关，仅查内部教材库，更快）")
    parser.add_argument("--chat", action="store_true",
                        help="多轮对话模式（带记忆：指代消解 + 长期用户记忆）")
    parser.add_argument("--pro", action="store_true",
                        help="用 deepseek-v4-pro（更强推理；默认 flash 更快）")
    args = parser.parse_args()

    graph = get_med_graph()
    model_tier = "pro" if args.pro else "flash"

    if args.chat:
        run_chat(graph, use_external=args.web, model_tier=model_tier)
        return

    if not args.question:
        parser.error("请给出问题，或用 --chat 进入多轮模式")

    # 单次模式：一次性 thread，不复用历史
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "user_id": "cli-user", "model_tier": model_tier}}
    result = graph.invoke({"question": args.question, "use_external": args.web}, config=config)

    mode = "内部教材 + 联网 Web" if args.web else "仅内部教材"
    print(f"\n[模式] {mode}")
    print(f"\nQ: {args.question}")
    print("规划子问题:")
    for sq in result.get("plan_questions", []):
        print(f"  - {sq}")
    _print_result(args.question, result, show_sources=True)
    print(f"\n{DISCLAIMER}")


if __name__ == "__main__":
    main()
