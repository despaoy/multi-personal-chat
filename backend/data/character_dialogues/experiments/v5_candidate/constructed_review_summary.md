# 阶段 3：150 条短构造复查材料摘要

生成时间：2026-08-23T03:32:57.508512+00:00｜数据源：KISAKI-CANONICAL-V4（sha256 已校验）

## 建议带分布（五维门禁，互不抵消；复查视角）

| 建议带 | 数量 |
|---|---:|
| prefer_keep | 134 |
| review_priority | 16 |
| prefer_exclude | 0 |

## 事实根基三态分布

| 状态 | 数量 | 含义 |
|---|---:|---|
| no_auto_flag | 142 | 自动未发现问题，仍需人工确认 |
| needs_human | 8 | 未引入原作人物（琉璃/夜子/理央）或元素，需人工判断 |
| auto_fail | 0 | 自动判定虚构（世界观/经历声称） |

- 技术关键词命中：0 条
- 多轮记录（需人工查一致性）：12 条
- 曾改写记录（历史审核含改写前文本，重点复查）：150 条
- 提问重复簇（≥2 条，仅提示）：3 簇
- 合计监督字符：2883（当前全部保留口径；复查后按决定重算）

## 复查重点

1. needs_human 记录多为角色问答类（问妃本人的关系/偏好/恐惧），assistant 提及琉璃/夜子/理央——需人工确认人物关系符合原作设定；
2. 曾改写记录（blindfix 子源 34 条含改写前文本）核对改写方向是否正确；
3. 多轮记录（12 条）检查时间/物品/立场一致性；
4. factual/事实与安全类（4 条）检查世界观事实声称（魔法之书等）。

## 人工复查方式

- 分批 Markdown（每批 ≤25 条，含完整对话/五维/历史审核/选择栏）：`constructed_review_batches/` 共 6 批
- revise = 需改写；决定阶段只接受 keep/exclude，revise 按 exclude 处理
- 决定文件：`constructed_review_decisions.json`，格式：
  `{"review_status": "approved", "reviewed_by": "owner", "decisions": {"<id>": "keep|exclude"}}`
  （150 个 ID 全覆盖、值域严格校验、review_status 必须为 approved）
- 勾选回收复用：`python scripts/collect_kisaki_v5_simulation_decisions.py --packet <constructed_review_packet.json> --batches-dir <constructed_review_batches> --output <constructed_review_decisions.json>`
