# P4 设计：+Reflection（反思回路）

## 目标

给医疗助手加 Agent 的 **Reflection 支柱**：作答后**自己审一遍**——当前证据够不够支撑一个
**安全、完整**的回答？不够就**用具体的缺失方向补检索、重新作答**，而不是把半截证据硬答完。

## 为什么医疗尤其需要反思

医疗答漏一个**关键安全方面**（只讲了副作用却没讲禁忌/相互作用）可能比答错更危险。
一次检索未必覆盖全部关键面 → 让模型答完后**自检遗漏**，针对性补检索，是把"完整性/安全性"
往上抬的低成本手段。

## 实现（`med_nodes.py: reflect / route_after_reflect / augment_retrieve`）

```
answer → reflect ──┬─ 充分 / 达上限 ──→ extract_memory（收尾）
                   └─ 不足 ──→ augment_retrieve ──→ fuse ──→ answer（重答）
```

| 环节 | 实现 | 关键点 |
|------|------|--------|
| 自检 | `reflect` 结构化输出 `ReflectResult{sufficient, reasoning, missing}` | prompt 严格：只在**缺关键信息**时判 false |
| 缺啥补啥 | `missing` 必须是**具体补充检索查询**（含药名/方面），不能是原问题复述 | 例 ✓「二甲双胍的禁忌症和用药注意」 ✗「还有什么要补充」 |
| 路由 | `route_after_reflect`：有 `missing_info` 且未达上限 → 补检索；否则收尾 | 条件边 |
| 补检索 | `augment_retrieve` 用 `missing_info` 再检索，**去重追加**到证据池（内/外两路） | 不覆盖已有证据，累积后回 `fuse` 重排 |
| 防死循环 | `MAX_REVISIONS=2` 硬上限 | 达上限即便仍"不足"也收尾，避免无限补检索 |

## 关键设计取舍

- **硬上限防死循环**：反思可能永远觉得"还能更全"，必须有上限；医疗取**克制**，默认最多补检索 1 次。
- **prompt 明确「不要因为还能更详尽就判 false」**：只有缺**关键**面才返工，避免无谓重答徒增延迟/成本。
- **证据累积而非覆盖**：`augment_retrieve` 把补到的证据**追加去重**进池子，和原证据一起重排——
  既不丢已有、又让新证据有机会浮上来。
- **`history` 统一在收尾节点 `extract_memory` 写**：反思会让 `answer` 执行多次，若在 `answer` 里写
  短期记忆会把同一轮的问与**中间草稿**重复记进去；故 `answer` 不写 history，由收尾节点记**最终**问答。
- **可关作对照**：`get_med_graph(use_reflect=False)` 退回「answer 直连 extract_memory」，
  便于 A/B 看反思带来的增益。

## 验证

问「二甲双胍能吃吗」（只问能否、未提背景）→ 首答偏泛 → `reflect` 判 false，
`missing`=「二甲双胍的禁忌症和肾功能要求」→ `augment_retrieve` 补到禁忌/GFR 阈值证据 →
重答补齐「肾功能不全（GFR<45）禁用」等关键安全信息，`revisions=2`。✓
（前端把 `revisions>1` 显示为「🔁 经过 N 轮证据反思核验」。）

## 待办

- 反思每轮一次额外 LLM 调用，增加延迟/成本 → 可只在"高风险问题"上触发反思。
- `MAX_REVISIONS` 固定为 2；可按问题复杂度/风险动态调。