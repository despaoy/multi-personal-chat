# 月社妃游戏文本场景切分（P3/P3.1/P3.2）

- 阶段：P3 保守场景切分与来源目录构建（P3.1：完整场景原文、temporal 语义、显式决定门禁、状态区分；P3.2：冻结事务原子性、low 审核包互斥重构、schema 门禁；P3.5：冻结前风险抽查，见 KISAKI_GAME_RAG_FREEZE_READINESS.md）
- 日期：2026-08-24
- 代码：`backend/knowledge/game_rag/story_units.py`（单元登记）、`backend/knowledge/game_rag/scene_segmenter.py`（检测/解析/验证/冻结/审核材料）
- 测试：`backend/tests/test_game_rag_scenes.py`（81 项）
- 审核材料：`backend/data/knowledge/tsukiyashiro_kisaki/scene_boundary_review/`（P3.2 重新生成）
- 决策确认：保守检测 + 最小 6 对话轮（用户确认）

## 1. 故事单元登记（18 单元 / 17 文件）

| 单元 | 卷号 | content_scope | 行范围 |
|---|---|---|---|
| vol01…vol13（16 个正篇单元，同卷号 6/8/9 各两个文件） | 1–13 | main_story | 各文件 1–EOF |
| epilogue_meta | None | promotional_meta | 日后谈 1–37 |
| epilogue_bonus | None | bonus_story | 日后谈 38–EOF（内容自 39 行起） |

- 《日后谈》固定边界取 P0 审计确认的第 38 空行（`EPILOGUE_FIXED_BOUNDARY=38`），是登记层的硬切分，不参与候选检测；
- `route=None`、`continuity_id=None`：不按文件名预设连续性（P0.2 决议）；
- `viewpoint=None`：视点为文本观察值且尚未逐卷人工核实，登记阶段不臆填，待边界抽检时一并补录；
- 任何段不得跨越固定单元边界（`split_segments_by_unit` 强校验）。

## 2. 边界候选检测（保守）

锚点语义：**锚点行号 = 新场景起始物理行**。候选按行去重（生效 > 不生效，high > medium > low）。

| 信号 | 置信 | 默认生效 | 判定 |
|---|---|---|---|
| time_jump | high | 是 | 叙述行首匹配明确时间跳跃词（次日/翌日/第二天/数日后/…/与此同时/另一方面/回忆的讲述…；词表扩展自既有 `SCENE_RESET_RE`） |
| time_of_day | medium | 是 | 叙述行首匹配时段/场景词（当天/当晚/那一夜/那天/放学后/清晨/傍晚/深夜/梦中…） |
| blank_line | medium | 是 | 相邻段行号间隔 ≥2（P2.1 已证明间隔只能来自空行；全语料 31 处候选） |
| transition_marker | high | 是 | **整行**分隔线（——/――/~~~/\*\*\*/===）或 ※ 注记行 |
| long_narration_gap | low | **否（仅记录）** | 单个叙述段连续 ≥5 行；需人工确认后经 adds 提升 |

关键修正（P3 开发中发现）：**行首 `——` 前缀行不是转场**。全语料实测 322 条破折号前缀行为行文破折号（如"——就这样，时间过去了。"），整行分隔线实为 0 条、※ 行仅 1 条（日后谈:39，属单元首行自动跳过）。初版前缀匹配造成 324 条误报，已改为整行匹配并附回归测试。

硬约束：

- **绝不在 dialogue 段内部锚点**（含未闭合台词吞并的叙述行，全语料测试断言 0 违例）；
- 只扫叙述段；叙述块内部的时间词行逐行检查（不受解析器叙述合并影响）；
- 单元首段之前不切分。

## 3. 两种边界解析模式（P3.1）

### 3.1 草稿预览模式 `plan_scene_boundaries`

自动生效候选 → 预览覆盖（remove/add）→ 最小轮数合并（<6 轮并入相邻）。仅供审核预览；人工 add 边界受保护不被自动合并撤销。

### 3.2 决定模式 `plan_scene_boundaries_from_decisions`

approved 冻结专用：边界**完全由决定生成**（`candidate_decisions` 中 boundary + `adds`），**不做任何自动合并**——每个边界都是人工显式确认的结果。

## 4. SceneDocument 组装（P3.1：完整场景原文）

`build_scene_documents(unit, segments, plan, *, source_lines)`：

- **text = 场景行号范围内的原文物理行逐字拼接**（LF 规范化后）：
  - dialogue 保留 `[说话人] 「台词」` 原文格式（P4 可判断具体台词归属）；
  - 多行台词只在第一行带标签（原文如此）；
  - 场景内部空行（自动合并产生的行号间隔）按原文行自动恢复；
  - 未闭合台词、重复闭引号（12青金石:2540 的 `」」`）原样保留；
- 全语料测试断言**每个 scene.text 与原文行范围逐字一致**（含 7 条未闭合台词、重复闭引号、妃 3 条跨行台词、合并空行边界的针对性覆盖）；
- `speakers` = 场景内说话人集合；`mentioned_characters`/`present_characters` 留空待 P4；
- **`temporal_scope=None`**（P3.1：尚未审核为 None；unknown 保留给"已审核但无法判断"）；
- `route=None`、`review_status=draft`；
- 场景 ID：`scene_ + sha256(source_path|scene|line_start|line_end)[:16]`（纯位置函数）。

## 5. 边界审核决定（P3.1 schema v2）

overrides 文档结构：

```json
{
  "schema_version": 2,
  "boundary_review_status": "draft | approved",
  "reviewer": "",
  "min_dialogue_turns": 6,
  "source_prefix": "gametext/纸上魔法使",
  "candidate_decisions": {"<unit_id>": {"<行号>": "boundary | no_boundary | null"}},
  "adds": {"<unit_id>": [行号]},
  "notes": ""
}
```

语义：

- **每个 high/medium 候选必须逐条显式决定**（boundary=保留边界 / no_boundary=不切分）；**null 即未决定，approved 验证拒绝**；
- **自动合并的 10 个候选同样是 high/medium 候选，必须明确决定**（no_boundary=确认合并，boundary=要求恢复边界），不得隐式接受；
- low 候选提升与人工新边界统一走 `adds`（add 行号必须可切分：段起始行或叙述内部行；dialogue 内部/空白行报错）。

`validate_boundary_overrides` 检查项（返回错误列表，approved 冻结前必须为空）：

1. `schema_version == 2`（P3.2：版本门禁，v1/缺失拒绝）；
2. `boundary_review_status == "approved"`；
3. reviewer 非空；
4. candidate_decisions / adds 覆盖全部 18 个登记单元且无未知单元；
5. 结构校验（P3.2）：candidate_decisions 及各单元决定必须是对象，adds 各项必须是数组，结构错误直接返回；
6. 所有 high/medium 候选均有合法决定；
7. 决定的行号确实属于该单元的 high/medium 候选集合；**行号必须为规范十进制**（`'01'`/`' 1'`/`'+1'` 拒绝——防止 `'01'` 与 `'1'` 绕过 JSON 键唯一性造成冲突决定）；
8. add 行号可切分且不在 dialogue/空白行内部，且必须为规范十进制；
9. 同一行不得同时出现在决定与 add 中（冲突）；
10. `min_dialogue_turns` 与审核包生成参数一致（提供 review_dir 时校验）。

## 6. 两种状态的区分（P3.1）

| 状态 | 含义 | 生效范围 |
|---|---|---|
| `boundary_review_status=approved`（overrides 文档） | **场景边界已冻结** | 仅边界；执行冻结后产出 scenes.jsonl |
| `SceneDocument.review_status=draft`（每条场景） | **人物、时间状态与知识内容尚未审核** | P4 审核通过并改为 approved 前，**这些场景不得进入正式索引** |

公开冻结入口 `freeze_reviewed_scenes(game_root, overrides_doc, review_doc, out_dir, *, review_dir=None)`：

- **从原始语料 + approved 决定重新解析并构建全部场景**，不接收任何现成场景列表（旧 `write_frozen_scenes(scenes, ...)` 已删除——防止把 draft 场景直接写入 frozen 文件）；
- 同时要求两份审核文档为 `approved`、reviewer 一致、low 必审范围无未决项、low boundary 与 replacement 全部准确并入 overrides、当前每个 oversized 场景均有结构化保留理由；
- 验证失败（缺审核文档/漏决定/未知单元/错误行号/理由缺失/冲突/min_turns 不一致/schema 版本错误等）立即抛错，不写出任何文件；
- 成功时经 `_freeze_pair_atomic` **两文件整体原子提交**（P3.2）：
  1. 先把两份内容全部写入各自 `.tmp`（写失败发生在提交前，旧文件未动）；
  2. 两份 tmp 均就绪后进入提交：备份旧 scenes → 替换 scenes.jsonl → 最后替换 boundary_manifest.json（manifest 作为本次冻结完成标志）；
  3. manifest 替换失败时**回滚 scenes.jsonl 到旧版本**（无旧版则删除新 scenes），不得留下"新 scenes + 旧 manifest"混合版本；故障注入测试覆盖四种场景（成功/二次替换失败有旧版/二次替换失败无旧版/阶段 1 写失败）。

## 7. 审核材料（P3.1 重新生成）

生成命令（从 backend 目录，幂等可再生）：

```text
py -3.10 -X utf8 -m knowledge.game_rag.scene_segmenter --game-root "..\gametext\纸上魔法使" --source-prefix "gametext/纸上魔法使" --out-dir "data\knowledge\tsukiyashiro_kisaki\scene_boundary_review"
```

审核策略（P3.1 + P3.2）：

- **必审清单：100% 覆盖全部 85 个 high/medium 候选**（DOS EOF 标记修复前为 86；逐条列出预览状态 + 触发文本 + 前后各 2 行原文摘录 + 决定填写位）；
- **oversized_scene_review**：51 个超过 300 行或 150 轮的场景，其**内部 567 条 low 候选逐条展示**（触发文本 + ±2 行原文摘录 + "提升为边界（写入 adds）"决定栏），优先审核；
- **其余 low 候选抽样（P3.2 互斥修正）**：仅从 672 − 567 = 105 条非 oversized 候选中分层抽样（单元×信号隔 5 取 1），两组互斥、数量守恒（有测试固化）；
- `overrides_template.json`：历史初始模板的待决定值**全部为 null**（禁止预填为通过），adds 为空数组。

人工流程：按必审清单逐条决定 → 对 oversized 场景评估其 low 候选（提升者写入 adds）→ 抽检其余 low 候选 → 两份审核文档分别批准 → 调用 `freeze_reviewed_scenes` 冻结（另行指令）。

## 8. 全语料结果（18 单元）

| 指标 | 值 |
|---|---|
| 段总数（输入） | 28,507（P2.1 输出，分配恰好一次） |
| dialogue / 妃台词 | 17,530 / 1,598（不变） |
| 候选总数 | 758（time_jump 16 / time_of_day 39 / blank_line 31 / long_narration_gap 672 / transition 0） |
| **必审候选（high/medium）** | **86**（预览生效 76 + 自动合并 10） |
| 预览场景数 | 94（决定模式全 boundary 时为 103——不做自动合并） |
| oversized 场景（>300 行或 >150 轮） | 51 |
| 场景文本 | 全部 94 个场景与原文行范围逐字一致（测试断言） |
| 确定性 | 候选/计划/场景/冻结产物两次运行完全一致 |

## 9. 已知限制与停止点

- 场景整体偏长是保守策略的预期结果；审计修正后仍有 24 个超限场景（均已在 `low_candidate_review.json` 的 `notes_by_unit.oversized_retained` 中逐条给出语义连续保留理由，见 P3.4）；
- time_jump 仅 16 条偏少：原作文风大量场景转换不加时间词（如 10黑曜石 926 行仅 1 条生效边界、2 个场景），漏切风险由人工审核兜底（P3.3 已补记多视角切换/嵌入文档类锚点）；
- 53 条审核范围外 low 候选保持显式 null（oversized 必审 579 + 分层抽检 36 + 顺带决定 3 + P3.5 决定 1 之外的其余候选，未审核即未决定；P3.5 已对 54 条原 null 做风险分层抽查 13 条，确认 1 处漏切并局部修复，详见 KISAKI_GAME_RAG_FREEZE_READINESS.md）；
- `speakers` 仅含说话人标签（14 位口径），无台词的在场人物需 P4；
- `viewpoint` 全部为 None 待人工补录；
- **本阶段到审核状态文件为止**：未冻结 scenes.jsonl、未生成任何事实/关系/事件卡、未修改游戏原文与既有 RAG 代码；`boundary_review_status` 保持 draft，等待外部复核。

## 修订记录

### P3.1（2026-08-24，按人工复核意见修正）

1. **完整场景原文**：`build_scene_documents` 新增 `source_lines` 参数，text 改为原文物理行逐字拼接（dialogue 保留说话人标签、空行按行号差恢复、未闭合/重复闭引号原样）；全语料逐字一致测试覆盖 94 场景及 7 条未闭合、1 条重复闭引号、妃 3 条跨行台词、合并空行边界。旧测试中 text 无标签的断言全部更新。
2. **temporal_scope 语义**：`StoryContext.temporal_scope` 改为 `TemporalScope | None` 默认 None（models.py + Schema 同步）；None=尚未审核，unknown=已审核但无法判断；P3 草稿场景使用 None；模型测试新增默认值与 roundtrip 覆盖。
3. **决定重构**：新增 schema v2（candidate_decisions + adds）与 `validate_boundary_overrides`；approved 不依赖"未填写即默认同意"——86 个 high/medium 候选（含 10 个自动合并项）必须逐条显式决定，null 拒绝；验证覆盖 reviewer/18 单元完备性/行号合法性/冲突/min_turns 一致性八项。
4. **状态区分与冻结重构**：删除 `write_frozen_scenes`（可绕过审核的旧入口）；新增从原始语料 + approved 决定重建场景的冻结实现，同时写出 scenes.jsonl（review_status=draft）与 boundary_manifest.json；P3.4 后该实现降为内部函数，公开入口必须经过两层审核。
5. **审核包重新生成**：必审清单 100% 覆盖 86 个 high/medium 候选；oversized_scene_review 收录 51 个超限场景并优先展示其 low 候选；其余 low 候选分层抽样；overrides 模板 86 个决定全部为 null。
6. **测试扩容**：31 → 64 项（+verbatim 类 5、决定模式 4、验证错误注入 13、冻结集成 6、全语料新增断言若干）。

### P3.2（2026-08-24，按人工复核意见修正）

1. **冻结事务原子性**：内部冻结实现经 `_freeze_pair_atomic` 两文件整体提交——先完整写两份 tmp，均成功后先替换 scenes 再替换 manifest（manifest 为完成标志）；manifest 替换失败回滚 scenes（有旧版恢复旧版、无旧版删除新版），不留"新 scenes + 旧 manifest"混合版本；故障注入测试覆盖成功/二次替换失败（有/无旧版）/阶段 1 写失败四种场景。
2. **low 审核包互斥重构**：oversized 场景内 567 条 low 候选逐条展示（触发文本 + ±2 行原文 + 提升决定栏）；抽样仅从其余 105 条（672−567）非 oversized 候选中分层抽取，两组互斥、数量守恒（测试断言 567+105=672 及逐单元集合关系）；boundary_stats 新增 oversized_low_candidate_count / remaining_low_candidate_count。
3. **schema v2 正式门禁**：`validate_boundary_overrides` 增加 schema_version==2 检查（v1/缺失拒绝，含冻结入口集成测试）；candidate_decisions/各单元决定必须是对象、adds 各项必须是数组（结构错误直接返回）；行号必须为规范十进制（`'01'`/`'007'` 拒绝——防止与 `'1'` 绕过 JSON 键唯一性造成冲突决定）。
4. **测试扩容**：64 → 81 项（+冻结原子性 4、schema 门禁 9、审核包互斥 3、旧验证测试补 schema_version）；P1/P2/P3 三套件合计 173 项全部通过。

### P3.3（2026-08-25，low 候选审核完成，待外部复核）

1. **结构化审核状态**：新增 `knowledge/game_rag/low_review.py` 与 `low_candidate_review.json`（schema v1）——`candidate_decisions` 覆盖全部 672 条 low 候选（boundary/no_boundary/null 三态，null=范围外未审核）、`oversized_scope`（579 条必审快照）、`external_sampling`（分层抽检 36 条 + 规则参数）、`replacement_adds`/`replacement_reasons`（候选本身不合适但附近存在准确锚点时的替代行及理由）、`notes_by_unit`（oversized 保留理由等）。校验门禁：单元集合完整、行号规范、候选真实属于 low、决定值合法、no_boundary 不与 adds 冲突、合并不破坏输入、临时文件原子写出。
2. **决定统计（P3.4 修正后）**：672 = boundary 13 + no_boundary 605 + null 54；覆盖口径 579（oversized 必审）+ 36（外部分层抽检）+ 3（顺带决定：vol01 L2545/L2594、vol02 L1101）+ 54（范围外 null）。
3. **合并结果（P3.4 修正后）**：`boundary_overrides.json` adds 共 194 条 = 13 条 low 候选直接提升 + 166 条替代锚点 + 15 条 high/medium 阶段人工新增；`boundary_review_status` 保持 draft。
4. **重算统计（P3.4 修正后）**：决定模式 259 场景，oversized 24 个，oversized 内残留 low 候选 134；24 个保留超限场景全部在 `notes_by_unit.oversized_retained` 记录语义连续理由。
5. **漏切修正**：补记 9 个非候选行锚点（vol01 L31 夜子→琉璃视角切换、vol02 L1102 红宝石概要入口、vol03 L1728/L1849/L2100/L2163 地点转换与视角切换、vol11 L1495、epilogue_bonus L105/L335 回顾节目讲述者切换）；修正 vol07 一处与 high/medium 决定冲突的替代锚点（改为修正原决定）。
6. **测试与守恒**：dialogue 17,530 / 妃台词 1,598 不变；0x1A 不进场景文本；日后谈尾部 dos_eof_marker warning 保留；连续运行确定一致；未执行正式冻结。

### P3.4（2026-08-26，长任务审计与修复）

1. **语义误切修正**：全上下文复核确认 vol12 L2053（萤石发光、妃随即出现）属于 L1976 起夜子连续意识场景，不存在时间、地点或视角转换；决定由 boundary 改为 no_boundary，并从 adds 删除。派生口径变为 259 场景、24 oversized、134 条 oversized 内 low 候选。
2. **low 审核契约强化**：每个直接 low boundary 必须有非空理由；replacement_adds 与 replacement_reasons 必须一一对应且拒绝孤儿理由；notes_by_unit 统一为对象；批准前要求当前全部 oversized 场景的范围与保留理由精确对应。清理了 vol06 L874 孤儿替代理由，并补录 vol12 L1976-2332 保留理由。
3. **不可绕过的冻结门禁**：原冻结函数降为内部 `_freeze_scenes_from_overrides`；唯一公开入口 `freeze_reviewed_scenes` 强制校验 low 审核文档存在且 approved、必审范围完整、两份 reviewer 一致、审核产生的 adds 与 overrides 精确一致，并在写盘前复算 oversized 理由覆盖。
4. **可追溯身份**：两份 draft 审核文档 reviewer 从泛化标签“人工审核者”改为稳定项目标识 `project_owner_01`；本轮未批准、未正式冻结。

### P3.5（2026-08-26，冻结前风险抽查与就绪验收）

1. **54 条 null 风险分层抽查**：按场景跨度/距边界距离/叙述信号/单元覆盖四层确定性抽取 13 条核验，11 条确认 null 正确；发现 vol12 L672（琉璃/彼方现实侧 → 克丽索贝莉露书房内第一人称视角切换）一处漏切——属对白先行冷开场、候选触发行 L670 本身为前段收束的孤立错误，按情况 B 局部修复（L670 决定 no_boundary + 替代锚点 672 入 replacement_adds/adds，含理由）。
2. **系统性模式评估**：识别「短叙述块 + 对白先行冷开场的视角/叙事层切换」盲区；vol12 L672 与 epilogue L758 两个已知实例均可通过人工非候选锚点修复，无需改变候选检测规则。
3. **外部复核修正**：P3.5 初稿错误地把“无候选”解释成“无法修复”，并遗漏审核状态中已明确记录的 vol04 L3315 教堂→图书馆尾声边界。外部复核将 L3315、L758 补入人工锚点并给出理由。
4. **最终派生口径**：672 = boundary 13 + no_boundary 606 + null 53；adds 197 = 13 直接提升 + 169 人工/替代锚点 + 15 high/medium 阶段人工新增；262 场景、oversized 24、oversized 内残留 low 133。
5. **门禁与测试**：冻结门禁继续保持；新增 L672 以及 L3315/L758 回归测试。两份审核状态保持 draft，未执行正式冻结。
