# 月社妃 Game RAG 场景切分 —— 冻结就绪与执行记录（P3.5/P3.6）

- 阶段：P3.5 冻结前风险抽查与就绪验收（2026-08-26）
- 审核人：`project_owner_01`
- 结论：P3.5 达到冻结就绪；项目负责人随后明确批准，P3.6 已完成正式冻结
- 上游文档：[KISAKI_GAME_RAG_SCENES.md](KISAKI_GAME_RAG_SCENES.md)（P3.1–P3.5 方法与全量审计记录）
- 本阶段边界：不重复 P3.4 已完成的 194 adds 全量审计、不逐条重审 24 个 oversized 场景、不强制审核全部 null；P3.6 仅执行批准、公开冻结入口和冻结后守恒验证

## 0. P3.6 正式冻结执行（2026-08-26）

- 授权：项目负责人明确批准继续冻结；审核人保持 `project_owner_01`；
- 状态：`boundary_review_status=approved`，low `review_status=approved`；
- 公开入口：`freeze_reviewed_scenes` 从原始 17 个 UTF-8 文本重新解析并生成正式双文件；
- 产物：`scenes.jsonl`（262 条）与 `boundary_manifest.json`（18 单元，`scene_review_status=draft`）；
- 文件 SHA256：scenes `10E48BEAF1A4922469BFBA268F3F95BCE2EF469BC42D447DF58A04FF6A867298`；manifest `5BA8636FFB94B15B6C5B136FC41BD4BE73EF87CB835F2C6F7A75B82BB1BB7A65`；
- P4A bundle SHA256：`b82d753eb93dbc1614b89e98c8498f03884c7d5dd433d223ef853708052c8595`；
- 两次冻结字节完全一致；262 个 scene id 唯一；17,530 dialogue / 1,598 妃台词 / 7 条未闭合引号 / 1 条 `dos_eof_marker` 守恒；场景文本无 `\x1A`；
- 冻结时发现并修复二次成功提交遗留 `scenes.jsonl.tmp.old` 的事务缺陷：成功提交现会清理备份，残留恢复副本会阻止覆盖；修复后无 `.tmp/.tmp.old`。

## 1. 当前基线（P3.5 修复后，从当前文件重读核实）

| 指标 | 数值 | 来源 |
| --- | --- | --- |
| high/medium 决定 | 47 boundary / 38 no_boundary（共 85 条） | boundary_overrides.json |
| low 决定 | 13 boundary / 606 no_boundary / 53 null（共 672 条） | low_candidate_review.json |
| adds（生效新增边界） | 197 = 13 直接提升 + 169 人工/替代锚点 + 15 high/medium 阶段人工新增 | boundary_overrides.json |
| 场景总数 | 262 | oversized_stats.json（决定口径重算） |
| oversized 场景 | 24（跨度 >300 行或对话 >150 轮） | oversized_stats.json |
| oversized 内残留 low 候选 | 133 | oversized_stats.json |
| dialogue / 妃台词 | 17,530 / 1,598（与 P3.4 一致，守恒） | 测试套件断言 |
| reviewer | 两份均为 `project_owner_01` | 两份审核文档 |
| 审核状态 | `boundary_review_status=approved`；low `review_status=approved` | 两份审核文档（P3.6） |
| 冻结产物 | `scenes.jsonl` / `boundary_manifest.json` 已生成 | P3.6 正式冻结 |

关键语义决定复核（P3.4 已定，本轮维持不变）：

- `vol07_7黑珍珠的求爱信号 L855 = boundary`
- `vol12_12青金石的幻想图书馆 L2053 = no_boundary`（萤石发光与妃出现属夜子连续意识场景，无时间、地点或视角转换，**未**恢复为 boundary）

## 2. 54 条 null 的风险分层方法

P3.4 收口时 54 条 null 全部位于**非 oversized 场景**（所属场景跨度最大 296 行，低于 300 行阈值），即均在 oversized 必审范围之外，语义风险天然低于已审范围。为判断其中是否隐藏系统性漏切，从 `low_candidate_review.json` 动态提取全部 null 候选，逐条计算：

- story unit、原文行号、候选触发文本；
- 当前所属场景范围、场景跨度（行数）、对话轮数；
- 距最近生效边界的距离；
- 候选所在叙述块长度；
- 叙述信号四类（仅用于分层排序，**不作为决定依据**）：
  - 时间词（次日/翌日/第二天/当天/当晚/清晨/黄昏/深夜等）；
  - 地点词（学校/教室/港口/海边/图书馆/书房/宅邸等）；
  - 视角指代（琉璃/汀/夜子/妃/岬/暗子/彼方等人名，鉴别力低，仅作参考）；
  - 嵌入叙事词（梦/回忆/追忆/书中/物语/日记/书信/幻/假想等）。

null 单元分布（修复前 54 条）：vol02×5、vol04×2、vol08怠惰×4、vol08时空×2、vol09白珍珠×3、vol09绿幽灵×5、vol10×3、vol12×12、vol13×8、epilogue×10。

## 3. 确定性抽样（13 条）

抽样规则（全部使用稳定排序与确定性规则，重复运行得到相同样本，无随机抽样；已入选 `external_sampling` 的候选本轮均未被重复选择——54 条 null 本身不在既往抽样范围内）：

- **层 1（场景跨度最大，4 条）**：按所属场景跨度降序取前 4 个场景（296/294/259/246 行），每场景取最接近场景中点的候选，每单元至多 1 条；
- **层 2（off-by-one 风险，3 条）**：距最近生效边界 ≤4 行的候选，按（距离, 单元, 行号）稳定排序取前 3；
- **层 3（叙述信号，3 条）**：时间/地点/嵌入信号数降序（视角指代因鉴别力低不计入排序），兼顾单元分散，按（信号数, 单元, 行号）稳定排序取前 3；
- **层 4（单元覆盖，3 条）**：从未入选单元中按（单元, 行号）稳定排序各取 1 条普通候选。

最终样本覆盖 10 个 story unit 中的 10 个（vol02、vol04、vol08怠惰、vol08时空、vol09白珍珠、vol09绿幽灵、vol10、vol12、vol13、epilogue），无单一单元占据样本主导（vol12 占 2/13）。

## 4. 逐条核验结论（决定 / 依据 / 置信度）

每条样本读取候选前后各 20 个物理行及所属场景起止摘要，必要时扩大到完整场景。核验判据：时间跳跃、地点转换、第一人称视角切换、现实↔回忆/梦境/假设/魔法重现切换、外框↔书中故事/日记/信件切换、新事件线开始；排除时间回指、总结句、排比列举、氛围描写、连续动作、独白内时间提及、对仗结构、同场景人物加入/离开。

| # | 层 | 单元 | 行 | 触发文本（截断） | 场景(跨度/轮) | 距边界 | 决定 | 依据（摘要） | 置信度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | vol12 青金石 | 629 | 从出生到现在，我的栖身之所… | L459-754 (296/94) | 126 | null 维持 | 克丽索贝莉露独白内身世回顾（时间回指），非切换点；本条场景级核验发现同场景 L670/L672 漏切，见 §5 | 高 |
| 2 | 1 | vol08 怠惰 | 485 | 滴溜溜的眼睛捕捉到了妃。 | L336-629 (294/132) | 146 | null 维持 | 连续动作与氛围描写，场景中段无切换信号 | 高 |
| 3 | 1 | vol09 绿幽灵 | 865 | 我露出了苦笑，以这句话收场。 | L758-1016 (259/119) | 107 | null 维持 | 对话收束句（总结句），后续延续同一对话场景 | 高 |
| 4 | 1 | epilogue | 747 | 琉璃和彼方都没有察觉喵。 | L632-877 (246/44) | 115 | null 维持 | 猫（萤）第一人称连续叙述；本条场景级核验发现 L758 检测盲区，见 §5/§12 | 高 |
| 5 | 2 | vol02 红宝石 | 750 | 我的第一个朋友，是一位品味… | L602-750 (149/58) | 1 | null 维持 | 场景末总结句；L751 已是生效边界，位置正确，无 off-by-one | 高 |
| 6 | 2 | vol09 绿幽灵 | 160 | 魔法使相信这才是夜子仅剩… | L1-160 (160/74) | 1 | null 维持 | 场景收束句；L161 边界位置正确，无 off-by-one | 高 |
| 7 | 2 | vol10 黑曜石 | 899 | 面对两人的死，我拿起了笔。 | L898-926 (29/4) | 1 | null 维持 | 暗子连续视角内动作承接（写书动机延续），L898 边界正确 | 高 |
| 8 | 3 | vol02 红宝石 | 820 | 这个认识过于危险可怕了。 | L751-823 (73/32) | 4 | null 维持 | 回忆叙述内部心理反应；时间/地点/嵌入词均为回指性提及 | 高 |
| 9 | 3 | vol12 青金石 | 2638 | 幻想图书馆前站着一个少女。 | L2553-2700 (148/51) | 85 | null 维持 | 终章同一场景人物出现（夜子现身，L2645 连续察觉），属同场景人物加入 | 中高 |
| 10 | 3 | vol09 白珍珠 | 1234 | 从最初开始到最后你都没能… | L1171-1260 (90/15) | 28 | null 维持 | 往事回忆叙述内部推进，地点/嵌入词均在同一回忆层内 | 高 |
| 11 | 4 | vol04 紫水晶 | 1116 | 可怎么回事？ | L1032-1190 (159/72) | 75 | null 维持 | 心理疑问句，连续叙事无切换 | 高 |
| 12 | 4 | vol08 时空 | 967 | 她以流畅的动作发起游戏。 | L809-1053 (245/99) | 87 | null 维持 | 连续动作描写，同一对局场景中段 | 高 |
| 13 | 4 | vol13 紫翠玉 | 589 | 岛上的人拿着武器包围教堂… | L563-763 (201/27) | 26 | null 维持 | 遊行寺往事回忆内部场景描写，回忆层内连续 | 高 |

结果：13 条抽样中 11 条为单纯确认 null 正确；样本 1（vol12 L629）与样本 4（epilogue L747）自身 null 决定经核验无误，但其场景级核验分别发现一处漏切与一处检测盲区（见 §5）。

### 情况 B 局部修复：vol12 L670 / 替代锚点 L672

对样本 1 所属场景 L459-754 做起止摘要核验时，发现该场景内部隐藏第一人称视角切换：

- L670「就算幻想图书馆遭忘去，我们仍记得一切」为琉璃/彼方现实侧决意收束（承接 L669 彼方「我们敲门吧」），**候选行本身非切换点**；
- L672「我无数次重复低语」起为克丽索贝莉露第一人称书房内视角（L675/L691 同源互证，L676「笨女孩。」、L677「妾咬着牙」视角明确），叙事由图书馆外转入书房内，至 L747-754 敲门声收束；
- 属对白先行冷开场：切换锚点在候选 L670 之后 2 行，不能机械上移候选行。

按情况 B 修复（孤立错误：触发行与切换点错位、且为 13 条抽样中唯一实例）：

- `candidate_decisions[vol12]["670"]`：null → `no_boundary`（附理由）；
- `replacement_adds[vol12]` 新增替代锚点 `672`（附 replacement reason，完整视角切换论证）；
- `boundary_overrides.json` 的 `adds[vol12]` 同步新增 672（194 → 195）；
- 场景 L459-754 拆分为 L459-671（琉璃/彼方现实侧）与 L672-754（克丽索贝莉露书房视角），场景总数 259 → 260。

情况 B 要求的同单元语义相近候选扩展检查：同场景其余 null（L642、L680、L719、L730、L744）随场景级核验一并确认 null 正确，未发现同类错误；未扩展到其他单元。

## 5. 系统性错误评估

发现一个**检测盲区模式**（非审核错误，而是候选生成规则的覆盖缺口）：

> 「短叙述块 + 对白先行冷开场的视角/叙事层切换」不会触发 `long_narration_gap` 候选——切换点本身不产生长叙述块，因此近旁若无其他候选，该切换永远不会进入 low 审核流程。

满足情况 C 的系统性判定条件（相同触发信号、相同语义错误原因、≥2 个不同位置）：

1. **vol12 L672-677**（琉璃/彼方 → 克丽索贝莉露视角）：因近旁恰有候选 L670，已通过 replacement anchor 修复（§4）；
2. **epilogue L758**（猫视角回忆 → 现实切换）：近旁无候选，但人工 add 本就允许记录非候选锚点；外部复核确认后已补入，不需要修改候选检测规则。

按情况 C 要求，仅扩大检查相同模式：对全部 54 条 null 候选执行同模式信号筛查（候选后 10 行窗口内的对白先行冷开场 + 强地点/时间确立词），除上述两处外未发现第三处同类漏切；筛查命中的其他信号（如 vol09 绿幽灵部分候选窗口）经核验为同场景内确立词回指，属筛查启发式的误报。**未扩大为全量重审。**

## 6. 审核范围扩展及理由

| 扩展动作 | 是否执行 | 理由 |
| --- | --- | --- |
| 情况 B：同单元语义相近 ≤5 条候选复查 | 是（同场景 5 条） | 规则要求；全部确认 null，佐证孤立性 |
| 情况 C：同模式候选筛查 | 是（全部 54 条 null 的窗口筛查） | 满足系统性判定，仅扩大到相同模式 |
| 全量 54 条 null 逐条审核 | 否 | 抽查 + 模式筛查未显示需要；避免无风险收益的扩大审核 |
| 全量原文重扫盲区 | 否 | 超出 P3.5 边界；盲区风险记入 §12 由外部裁量 |
| 其他单元扩展 | 否 | 情况 B/C 均未显示跨单元扩散 |

## 7. 实际修改文件

| 文件 | 修改内容 |
| --- | --- |
| `backend/data/knowledge/tsukiyashiro_kisaki/scene_boundary_review/low_candidate_review.json` | L670 决定 null→no_boundary + reason；`replacement_adds[vol12]` +672 + replacement_reason |
| `backend/data/knowledge/tsukiyashiro_kisaki/scene_boundary_review/boundary_overrides.json` | `adds[vol12]` +672；外部复核补入 vol04 L3315、epilogue L758（最终 197） |
| `backend/data/knowledge/tsukiyashiro_kisaki/scene_boundary_review/oversized_stats.json` | 从真实决定重新生成（262 场景；非手工改数） |
| `backend/tests/test_game_rag_scenes.py` | 新增 L672 回归与 L3315/L758 已知非候选切换回归；同步真实统计断言 |
| `docs/research/KISAKI_GAME_RAG_SCENES.md` | P3.5 修订记录 |
| `docs/research/KISAKI_GAME_RAG_FREEZE_READINESS.md` | 本报告（新增） |

未修改：游戏原文、解析器、切分阈值、low 候选检测规则、SFT/实验数据、通用 RAG 模块、既有 166 条 replacement、approved/null 语义。临时脚本（null 清单提取、修复、门禁、模式筛查共 5 个）任务结束前已全部删除。

## 8. 修改前后统计

| 指标 | P3.4 基线 | P3.5 后 | 变化 |
| --- | --- | --- | --- |
| low boundary | 13 | 13 | — |
| low no_boundary | 605 | 606 | +1（L670） |
| low null | 54 | 53 | −1（L670） |
| low 总数 | 672 | 672 | — |
| adds | 194 | 197 | +3（L672、L3315、L758） |
| 场景总数 | 259 | 262 | +3（三处明确切换均已拆分） |
| oversized 场景 | 24 | 24 | — |
| oversized 内残留 low | 134 | 133 | −1（vol04 尾声从 oversized 场景拆出） |
| dialogue / 妃台词 | 17,530 / 1,598 | 17,530 / 1,598 | —（守恒） |
| high/medium 决定 | 85（47/38） | 85（47/38） | — |

全部统计由修复脚本从真实决定重算（oversized_stats.json 为决定口径重建），无手工修改数字。

## 9. 自动门禁验证（13 项，全部 PASS）

| # | 门禁 | 结果 |
| --- | --- | --- |
| 1 | boundary_overrides / low_candidate_review / oversized_stats 均可正常加载 | PASS |
| 2 | 两份 reviewer 均为 `project_owner_01` | PASS |
| 3 | 两份审核状态均为 draft | PASS |
| 4 | low 决定总数 672 = 13 boundary / 606 no_boundary / 53 null | PASS |
| 5 | adds(197) = direct(13) + replacement/manual(169) + legacy(15) 闭合；direct∪replacement ⊆ adds；无 no_boundary 冲突 | PASS |
| 6 | 13 条 direct low boundary 均有理由（13/13） | PASS |
| 7 | replacement_adds(169) 与 replacement_reasons(169) 一一对应且非空 | PASS |
| 8 | oversized 场景(24) 与 oversized_retained 保留理由精确对应（span/turns/理由 24/24） | PASS |
| 9 | `freeze_reviewed_scenes` 是唯一公开冻结入口（包导出 / low_review / segmenter 三层一致） | PASS |
| 10 | 包入口不暴露 `freeze_scenes` / `_freeze_scenes_from_overrides` | PASS |
| 11 | draft 状态无法冻结（公开入口拒绝：`ValueError: low review_status must be approved, got 'draft'`） | PASS |
| 12 | 失败路径不写出任何冻结文件 | PASS |
| 13 | 无 `scenes.jsonl` / `boundary_manifest.json` 残留 | PASS |

## 10. Ruff / 格式 / pytest

自 `backend` 目录运行：

```text
py -3.10 -m ruff check knowledge/game_rag tests/test_game_rag_models.py tests/test_game_rag_parser.py tests/test_game_rag_scenes.py tests/test_game_rag_low_review.py
→ All checks passed!

py -3.10 -m ruff format --check knowledge/game_rag tests/test_game_rag_models.py tests/test_game_rag_parser.py tests/test_game_rag_scenes.py tests/test_game_rag_low_review.py
→ 10 files already formatted

py -3.10 -m pytest tests/test_game_rag_models.py tests/test_game_rag_parser.py tests/test_game_rag_scenes.py tests/test_game_rag_low_review.py -q
→ 225 passed in 25.72s
```

守恒验证由既有测试覆盖并全部通过：dialogue 17,530；妃台词 1,598；解析段完整分配；场景无倒序/重叠/遗漏；7 条未闭合引号告警保持；9 条合法跨行台词保持；`\x1A` 不进入场景文本；日后谈尾部 `dos_eof_marker` 保持；连续运行结果确定一致。未为通过测试放宽任何门禁或删除断言。

## 11. 冻结就绪结论

**P3.5 达到冻结就绪，P3.6 已按授权执行冻结。** 冻结前依据：

1. P3.4 全量审计（194 adds 逐条复核、24 oversized 逐场景保留理由）已完成且本轮门禁复核闭合；
2. 54 条范围外 null 经四层风险抽查与同模式筛查；vol12 L672、epilogue L758 两个已确认实例均已修复；外部复核同时落实审核状态中早已明确记录的 vol04 L3315 教堂→图书馆尾声边界；
3. 审核数据、派生统计与公开冻结门禁三者一致（13 项门禁全 PASS）；
4. 唯一公开冻结入口 `freeze_reviewed_scenes` 在 draft 状态下被正确拒绝，冻结事务未被触碰；
5. 守恒指标全部不变，测试 224 项全通过。

上述冻结路径已完成；下一阶段由 P4B 对冻结场景生成模型候选，人工复核后才可进入 approved enriched 输出。

## 12. 剩余风险

| # | 风险 | 评估 | 缓解 |
| --- | --- | --- | --- |
| 1 | 「短叙述块 + 对白先行冷开场」盲区可能存在其他未发现实例 | 发生率低；本轮已知的 vol12 L672 与 epilogue L758 均已落实人工锚点 | 保留为抽样残余风险，不再扩大成全文重审 |
| 2 | 53 条 null 保持未决定（范围外） | 13 条分层样本自身决定均正确；null 明确表示未审核未决定 | 冻结语义允许 null；后续仅在出现新证据时追加审核 |
| 3 | vol12 场景 L459-671 仍偏长 | 低于 oversized 阈值且视角统一 | 无需继续切分 |

---

**P3.6 完成确认**：风险抽查与情况 B/C 检查保持有效；两份审核状态已批准；正式冻结双文件通过 P4A 输入门禁与确定性验证；未运行真实 LLM、未生成候选或 enriched 产物、未提交/推送/连接服务器。
