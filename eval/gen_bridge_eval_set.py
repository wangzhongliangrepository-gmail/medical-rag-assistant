"""多跳「桥接」评测集生成工具：造需要两跳才能答的题，金标是**一对**有依赖关系的 chunk。

为什么要它：答案层反思 on/off 消融（见 docs/EVAL.md §6）证明——反思的收益集中在「首检索
不够」的难题上，但现有金标全是**单跳单 chunk**，价值被均值冲平。桥接题填这个缺口：
  问「X 病的一线药有什么副作用」——chunk A 指出一线药是 Y，chunk B 才有 Y 的副作用，
  **单看 A 答不出来**，必须 A→B 两跳。这正是反思/链式精化能发力的场景。

生成流程（半自动，产出草稿供人工审）：
  ① 抽种子 chunk A → LLM 判断 A 是否明确指认「某病/情况 → 某药/疗法 E」并抽出 description/entity/aspect
  ② 用 search(f"{E}的{aspect}") 找第二跳 B（source 与 A 不同、且正文含 E，确保 B 真展开了 E）
  ③ LLM 据 description+aspect 出题，**不出现药名 E**（逼出 A→B 两跳），自检单段不可独答
  ④ 金标 = [[A_src,A_chunk],[B_src,B_chunk]]，附两段原文供人工审

产出 JSON（字段兼容 eval_retrieval.py 的 gold 与 eval_answer.py 的 question）：
  [{"question","gold":[[s,c],[s,c]],"bridge_entity","hop":"bridge","_chunk_a","_chunk_b"}]

用法（从仓库根运行；需 Xinference + DeepSeek + Qdrant 在线）：
  python eval/gen_bridge_eval_set.py --n 12                 # 目标产出 12 条桥接题
  python eval/gen_bridge_eval_set.py --n 12 --max-probes 80 # 最多探查 80 个种子 chunk
  python eval/gen_bridge_eval_set.py --pro                  # 用 deepseek-v4-pro 出题（更稳）
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根入 path
import _bootstrap  # noqa: F401  必须最先导入：放行 OpenMP 重复加载
import argparse
import json
import random

from pydantic import BaseModel

from config import LLM_FLASH, LLM_PRO, QDRANT_COLLECTION
from kb_search import search
from llm import get_llm
from vectordb import get_client

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，强制 UTF-8 防 emoji 编码崩
except Exception:
    pass

_ASPECTS = {"副作用", "禁忌", "作用机制", "药物相互作用"}
# 第二跳 B 不仅要含实体名，还要确实在讲该「方面」，否则会选到「机制段/别的药段」等歪 B。
_ASPECT_KEYWORDS = {
    "副作用": ("副作用", "不良反应", "毒性"),
    "禁忌": ("禁忌", "禁用", "慎用"),
    "作用机制": ("作用机制", "机制", "药理作用"),
    "药物相互作用": ("相互作用", "合用", "联用", "并用"),
}
# 种子预筛：教材覆盖极广，「病→药一线指认」稀少，随机抽命中率≈0。
# 先只保留含治疗推荐标记的段（库中约 979 段），再 probe，命中率大幅提升。
_SEED_MARKERS = ("一线", "首选", "治疗首选", "推荐使用", "金标准")

_PROBE_PROMPT = """下面是一段中文医学教材。判断它是否**明确指认了某种疾病/情况所对应的具体药物或疗法**
（形如「X 的一线用药是 Y」「治疗 X 首选 Y」「X 常用 Y 治疗」），且该 Y 还有值得进一步追问、
但本段未充分展开的方面（副作用 / 禁忌 / 作用机制 / 药物相互作用）。

- 若是：has_bridge=true；description 填那个**疾病/情况**（务必**不含药名 Y**，用于指代）；
  entity 填被指认的具体药名/疗法 Y；aspect 从「副作用 / 禁忌 / 作用机制 / 药物相互作用」中选一个最自然的。
- 若否（没有这种「病→药」的明确指认，或只是泛泛而谈）：has_bridge=false，其余留空。

教材段落：
{text}"""

_QGEN_PROMPT = """构造一道**需要两跳才能回答**的中文医疗检索题。

已知：
- 某疾病/情况：「{description}」
- 它对应的药物/疗法是 E（E 的真实名称是「{entity}」，但**问题里绝对不能出现 E 或其明显同义词/简称**）
- 要追问的方面：E 的「{aspect}」

请生成一个自然、口语化的问题，满足：
- 用「{description}」来指代那个病/情况，让读者**必须先查出 E 是什么**，才能接着查 E 的「{aspect}」；
- 落点在「{aspect}」上；
- **不得出现药名「{entity}」**。
- 示例：description=「2 型糖尿病」、entity=「二甲双胍」、aspect=「副作用」
  → "2 型糖尿病的一线治疗药物有哪些副作用？"

并判断 single_chunk_answerable：是否存在单独一段教材就能独立答完此题（正常的桥接题应为 false）。"""


class BridgeProbe(BaseModel):
    has_bridge: bool
    description: str = ""
    entity: str = ""
    aspect: str = ""


class BridgeQuestion(BaseModel):
    question: str
    single_chunk_answerable: bool


def iter_chunks(client):
    """翻页遍历集合，产出每个 chunk 的 payload（不取向量）。"""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=QDRANT_COLLECTION, limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            yield p.payload
        if offset is None:
            break


def find_hop_b(entity: str, aspect: str, a_source_id) -> dict | None:
    """检索第二跳 B：与 A 不同来源、正文含 entity、且确实在讲该 aspect（含方面关键词）。"""
    kws = _ASPECT_KEYWORDS.get(aspect, (aspect,))
    for cand in search(f"{entity}的{aspect}", top_k=5):
        t = cand["text"]
        if cand["source_id"] != a_source_id and entity in t and any(k in t for k in kws):
            return cand
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12, help="目标产出桥接题数")
    p.add_argument("--max-probes", type=int, default=80, help="最多探查多少个种子 chunk")
    p.add_argument("--min-len", type=int, default=120, help="种子 chunk 最小长度")
    p.add_argument("--pro", action="store_true", help="用 deepseek-v4-pro 出题（更稳）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "eval_set_bridge_draft.json"))
    args = p.parse_args()

    client = get_client()
    print(f"[1/3] 遍历集合 '{QDRANT_COLLECTION}' 收集种子 chunk...")
    pool = [c for c in iter_chunks(client)
            if c.get("text") and len(c["text"]) >= args.min_len
            and any(m in c["text"] for m in _SEED_MARKERS)]
    if not pool:
        print("（没有含治疗推荐标记的 chunk：检查库，或放宽 _SEED_MARKERS）")
        return
    random.seed(args.seed)
    random.shuffle(pool)
    print(f"      含治疗标记的种子 {len(pool)} 个；目标 {args.n} 条，最多探查 {args.max_probes} 个")

    llm = get_llm(model=LLM_PRO if args.pro else LLM_FLASH)
    probe = llm.with_structured_output(BridgeProbe, method="json_mode")
    qgen = llm.with_structured_output(BridgeQuestion, method="json_mode")

    rows, probes = [], 0
    n_bridge = n_bfound = 0  # 诊断：识别出「病→药」的数 / 找到第二跳 B 的数
    for a in pool:
        if len(rows) >= args.n or probes >= args.max_probes:
            break
        probes += 1
        try:
            pr = probe.invoke([
                ("system", '只输出合法 JSON：{"has_bridge":bool,"description":"...","entity":"...","aspect":"..."}'),
                ("human", _PROBE_PROMPT.format(text=a["text"])),
            ])
        except Exception:
            continue
        if not (pr.has_bridge and pr.description and pr.entity and pr.aspect in _ASPECTS):
            continue
        # 描述里若已暴露实体名，桥接失效，跳过
        if pr.entity in pr.description:
            continue
        n_bridge += 1
        b = find_hop_b(pr.entity, pr.aspect, a["source_id"])
        if b is None:
            continue
        n_bfound += 1
        try:
            qg = qgen.invoke([
                ("system", '只输出合法 JSON：{"question":"...","single_chunk_answerable":bool}'),
                ("human", _QGEN_PROMPT.format(description=pr.description, entity=pr.entity, aspect=pr.aspect)),
            ])
        except Exception:
            continue
        q = (qg.question or "").strip()
        # 质量闸：要有问题、单段不可独答、且问题里没漏出药名
        if not q or qg.single_chunk_answerable or pr.entity in q:
            continue
        rows.append({
            "question": q,
            "gold": [[a["source_id"], a["chunk_id"]], [b["source_id"], b["chunk_id"]]],
            "bridge_entity": pr.entity,
            "aspect": pr.aspect,
            "hop": "bridge",
            "_chunk_a": a["text"][:400],
            "_chunk_b": b["text"][:400],
        })
        print(f"   ✓ [{len(rows)}/{args.n}] (探查{probes}) {q[:42]}  桥接实体={pr.entity}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n[3/3] 完成：{len(rows)} 条桥接题（探查 {probes} 个种子）→ {args.out}")
    print(f"      漏斗：探查 {probes} → 识别病→药 {n_bridge} → 找到第二跳B {n_bfound} → 出题通过 {len(rows)}")
    print("      [注意] 草稿！人工审：1)问题是否真要两跳 2)A/B 两段是否分别承担一跳 3)有没有漏出药名。")
    print("      审核后：检索评测 python eval/eval_retrieval.py --gold " + args.out + " --match chunk")
    print("              反思消融 python eval/eval_answer.py --gold " + args.out + "  （再 --no-reflect 跑对照）")


if __name__ == "__main__":
    main()