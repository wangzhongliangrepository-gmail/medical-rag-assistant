# 图结构演进：M1 → M3

根据 README 结果表与当前代码，记录学习助手 Agent 从单跳基线到反思回路的图结构演进。

## M1：单跳 RAG 基线

```
question → retrieve(BGE 重排 top-3) → answer → END
```

最简单的对照基线。10 段 context 直接重排取 top-3，DeepSeek 一次性看全部证据作答。

**bridge 0.5241 / comparison 0.7059 / 总 0.5550 EM**

---

## M2-static：+Planning（静态子问题）— ❌ 未超越 M1

```
question → planner(一次拆完所有子问题) → retrieve → answer → END
                     │
              SubQ1, SubQ2 同时生成，互不依赖
```

SubQ2 仍是泛指（"the actress who played..."），证据池没变，等于换了查询词。comparison 还跌了 0.18。

**bridge 0.5241 / comparison 0.5294 / 总 0.5250 EM**

---

## M2-refine：+Planning（链式精化）— ✅ bridge +0.036

```
question → planner → retrieve ──→ refine ──┐
                        ▲          (提取中间答案,    │ 还有子问题
                        │           精化下一跳查询)  │
                        └──────────────────────────┘
                                   │ 子问题用完
                                   ▼
                                 answer → END
```

refine 把 SubQ1 检索出的中间答案（"Shirley Temple"）代入 SubQ2，再检索。

**bridge 0.5602 / comparison 0.5882 / 总 0.5650 EM**（当前锁定基线）

---

## M3：+Reflection（当前代码，含 classify 分流）

```
                  ┌─ comparison ─→ retrieve → refine → answer → END
question → classify┤                  ▲          │
                  └─ bridge ─→ planner─┘          │（子问题循环同 M2）
                                                   ▼
                                                 answer
                                                   │
                                              [bridge only]
                                                   ▼
                                      reflect(证据够不够?)
                                       │              │
                                  够 → END      不够 → prepare_retry
                                                       │（missing_info 入队）
                                                       ▼
                                                   retrieve（重检索，≤3 轮）
```

两条关键设计：

- **classify 分流**：comparison 走 M1 式短路径（避免 M2 那次 -0.18 回归），bridge 走完整链式+反思。实测 classify 准确率 96%（bridge 95.8% / comparison 97.1%，n=200）。
- **Reflexion 回路**：bridge 答完后 reflect 自判证据是否充分，不够则把 `missing_info` 当新查询重检索，硬上限 `MAX_REVISIONS=3`。

**当前问题**：M3 bridge EM=0.5181，反而比 M2-refine 的 0.5602 回归了 4.2 个点，`avg_revisions=0.83` 说明反思几乎没产生有效 retry，纯增开销。已加 `--no-reflect` ablation 开关（`eval.py --no-reflect` / `get_graph(use_reflect=False)`），用于拆开「classify+refine」与「反思回路」各自的贡献。

README 结果表中 M3 行待 ablation 对照实验跑完后填入。

## 结果汇总表

| 阶段 | 配置 | EM | F1 |
|------|------|----|----|
| M1 | 单跳 RAG 基线（top-3，n=200） | 0.5550 | 0.6971 |
| M1 bridge | └─ bridge 子集（n=166） | 0.5241 | 0.6759 |
| M1 comparison | └─ comparison 子集（n=34） | 0.7059 | 0.8006 |
| M2-static | +Planning 静态子问题（n=200） | 0.5250 | 0.6656 |
| M2-refine | +Planning 链式精化（n=200） | 0.5650 | 0.7024 |
| M2-refine bridge | └─ bridge 子集（n=166） | 0.5602 | 0.7065 |
| M2-refine comparison | └─ comparison 子集（n=34） | 0.5882 | 0.6823 |
| M3 | +Reflection | 待填 | 待填 |
| M3 bridge | └─ bridge 子集（初测 EM=0.5181，回归中） | — | — |
| M4 | +Memory | 待填 | 待填 |

---

# 概念词典：Planning / refine / retrieve / Reflection / reflect / SubQ1 / SubQ2

用一个具体例子串起来。假设用户问：

> **"演 Corliss Archer 的女演员，担任过什么政府职位？"**

这题答不了一步——你得先知道"谁演的 Corliss Archer"，再去查"那个人的政府职位"。这种"答案藏在两段不同资料里、要接力"的题叫 **bridge（桥接）题**。下面这些概念就是为了拆这种题。

## 1. Planning（规划）— 一个"支柱/能力"

Agent 四大支柱之一。指**让模型先想清楚要分几步、每步查什么，而不是闷头一次性作答**。在本项目里，Planning 这个能力由 `planner` 节点实现。

## 2. SubQ1 / SubQ2（子问题 1 / 子问题 2）

planner 把原问题**拆成的小问题**，SubQ = Sub-Question。上面那题会被拆成：

- **SubQ1**：`"谁在 Kiss and Tell 里演了 Corliss Archer？"`
- **SubQ2**：`"那位女演员担任过什么政府职位？"`

编号就是执行顺序——先答 SubQ1，它的答案是答 SubQ2 的前提。

## 3. retrieve（检索）— 一个"节点/动作"

**retrieve = 拿一个问题去资料里找最相关的段落**。本项目里它干两件事：

- 用 BGE 重排模型，把 10 段 context 按和当前子问题的相关度打分
- 取最相关的 top-3 段，作为"证据（evidence）"

注意：retrieve 一次只处理**一个**子问题。先 retrieve(SubQ1)，之后再 retrieve(SubQ2)。

## 4. refine（精化）— 一个"节点/动作"

这是本项目的**关键改进**。问题在于 SubQ2 本身是"泛指"的——`"那位女演员"`到底是谁？模型不知道，拿这句去检索等于白查。

**refine 节点的作用：在两次 retrieve 之间，把上一跳的答案填进下一个子问题。**

```
retrieve(SubQ1) → 找到证据，里面写着 "Shirley Temple"
       ↓
refine：从证据里提取中间答案 "Shirley Temple"，
        把 SubQ2 改写成 → "Shirley Temple 担任过什么政府职位？"
       ↓
retrieve(精化后的 SubQ2) → 这下能精准查到 "美国驻加纳大使"
```

没有 refine（M2-static），SubQ2 一直是泛指，检索捞不到对的段，所以分数不涨。加了 refine（M2-refine），bridge EM +0.036。这就是"链式精化"。

## 5. Reflection（反思）— 又一个"支柱/能力"

Agent 四大支柱之一。指**模型答完后，回头检查自己的答案/证据够不够、有没有瞎编**。本项目里由 `reflect` 节点实现。

## 6. reflect（反思节点）— 一个"节点/动作"

具体动作：**答完之后，让模型当"质检员"判断——现有证据能不能支撑这个答案？**

- **够了**（sufficient=true）→ 结束，输出答案
- **不够**（sufficient=false）→ 它要写一个具体的补充检索查询（`missing_info`），然后回到 retrieve **重新检索一轮**，最多重试 `MAX_REVISIONS=3` 次

理论上这能救回"第一次没查全"的题。但实测：reflect 频繁触发重试（avg_revisions≈0.9），却没改变对错——这正是待排查的问题。

## 串起来看（M3 bridge 完整流程）

```
原问题
  │
classify  ── 判断这是 bridge 题（要接力）还是 comparison 题（对比）
  │
planner   ── 拆成 SubQ1、SubQ2                          【Planning 支柱】
  │
retrieve  ── 拿 SubQ1 去 10 段里检索，找到证据
  │
refine    ── 从证据提取 "Shirley Temple"，精化 SubQ2
  │
retrieve  ── 拿精化后的 SubQ2 再检索，找到政府职位的证据
  │
answer    ── 综合所有证据，生成最终答案
  │
reflect   ── 质检：证据够吗？不够就回 retrieve 再来一轮   【Reflection 支柱】
  │
最终答案 + 引用
```

| 名字 | 类别 | 一句话作用 |
|------|------|-----------|
| **Planning** | 支柱（能力） | 先拆步骤再行动 |
| **SubQ1/SubQ2** | 数据 | planner 拆出的有序子问题 |
| **retrieve** | 节点（动作） | 拿一个子问题去资料里检索 top-3 证据 |
| **refine** | 节点（动作） | 把上一跳答案填进下一个子问题，让检索更精准 |
| **Reflection** | 支柱（能力） | 答完回头自查够不够 |
| **reflect** | 节点（动作） | 质检证据，不够就触发重检索 |

简历叙事里，**Planning 和 Reflection 是"四大支柱"层面的卖点**，而 retrieve / refine / reflect 是把这些支柱落到 LangGraph 图里的**具体节点**。
