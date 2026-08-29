# 月社妃游戏 RAG 数据模型 Schema（P1）

- 阶段：P1 统一数据模型
- 日期：2026-08-24
- 代码位置：`backend/knowledge/game_rag/models.py`
- 测试：`backend/tests/test_game_rag_models.py`（33 项，全部通过）
- 上游依据：`KISAKI_GAME_RAG_BASELINE_AUDIT.md`（P0/P0.1/P0.2 修订后）

> 本文是 P1 阶段的数据契约记录。阶段号只用于研究过程追溯；当前生产功能名、
> 索引和运行入口见[多粒度角色知识检索](../architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)。

本阶段只定义数据契约，不包含解析、切分、向量化、数据库写入或 API 集成。

---

## 1. 模型职责

### 1.1 枚举

| 枚举 | 职责 | 取值 |
|---|---|---|
| `SegmentType` | P2 解析输出的基本类型 | `dialogue` / `narration` |
| `QuoteStyle` | 台词引号样式（叙述为 `none`） | `corner`（「」）/ `double_corner`（『』）/ `curly`（""）/ `none` |
| `ContentScope` | 内容类型维度 | `main_story` / `bonus_story` / `promotional_meta` / `unknown` |
| `TemporalScope` | 时间状态维度 | `current` / `flashback` / `reconstruction` / `hypothetical` / `unknown` |
| `RouteType` | 路线维度（**不含 flashback**） | `main` / `branch` / `parallel` / `unknown` |
| `RealityStatus` | 内容现实/可靠性状态 | `objective` / `character_claim` / `inferred` / `fictional` / `conflicted` / `unknown` |
| `ReviewStatus` | 人工审核状态 | `draft` / `needs_review` / `approved` / `rejected` |
| `DocumentType` | 知识库文档类型 | `scene` / `fact` / `relation` / `event` / `chapter_summary` |

### 1.2 结构模型（组合复用，无继承体系）

| 模型 | 职责 |
|---|---|
| `SourceSpan` | 原文出处：`source_path + line_start + line_end`（行号从 1 起，`line_end >= line_start`；本阶段不读取文件、不校验路径存在性） |
| `StoryContext` | 故事结构上下文：四个结构维度字段的载体，随各文档组合使用 |
| `ScriptSegment` | **P2 解析器的输出契约**：一条对话或一段叙述，携带告警列表 |
| `SceneDocument` | 场景文档（P3 切分产物）：完整场景原文 + 人物在场信息 |
| `FactDocument` | 事实卡：`subject-predicate-value` 三元组，一条记录只表达一个事实 |
| `RelationDocument` | 关系卡：`subject-relation-target` 及证据 |
| `EventDocument` | 事件卡：`participants / causes / outcomes` |
| `ChapterSummaryDocument` | 章节摘要：只服务跨章节全局问题，不能作为唯一事实证据 |

所有模型 `extra="forbid"`（拒绝未知字段），关键内容字符串（text / evidence_text / subject / target / summary 等）去空白后不得为空；`dialogue` 必须有 `speaker`，`narration` 不强制。

---

## 2. 四个结构维度的区别

P0.2 审计修正后确立的语义维度分离原则：**内容类型、时间状态、文件间结构关系是三个独立维度，不得混用**。

| 维度 | 字段/枚举 | 回答的问题 | 示例 |
|---|---|---|---|
| 内容类型 | `content_scope: ContentScope` | 这段文本属于哪类内容？ | 日后谈第 1–37 行 → `promotional_meta`；第 39 行起《萤色光景》→ `bonus_story`；正篇 → `main_story` |
| 时间状态 | `temporal_scope: TemporalScope` | 叙述处于哪个叙事时间层？ | 妃的回忆 → `flashback`；魔法重现的角色 → `reconstruction` |
| 路线 | `route: RouteType \| None` | 这个故事单元在文件间结构关系中是主线/分支/平行？ | 剧情结构审核后才能填写 |
| 连续性 | `continuity_id: str \| None` | 哪些故事单元处于同一连续时间线？ | 剧情结构审核后才能填写 |

要点：

- **`flashback` 只属于 `TemporalScope`**。回忆是时间状态，不是路线；把时间状态混入路线维度会导致后续检索过滤混乱（测试断言：`RouteType("flashback")` 必须抛错）。
- 同编号多文件（6/8/9 章）的关系仅由 `volume_number + story_unit_id` 表达，**不预设** `6A/6B` 之类的路线编号。
- **本阶段不宣称剧情路线已确定**。`route` 与 `continuity_id` 默认 `None`，只有完成剧情结构人工审核后才填写；在此之前 P3 场景切分不得跨同编号文件合并证据。

## 3. `None` 与 `unknown` 的区别

以 `route` 为例（`continuity_id`、`temporal_scope` 同理）：

| 值 | 语义 | 后续处理 |
|---|---|---|
| `None` | **尚未审核**（默认值） | 待剧情结构审核；审核前不得当作过滤条件 |
| `RouteType.unknown` | **已审核但无法确定** | 审核流程的显式产出，可作为过滤条件（匹配"无法确定路线"的文档） |

`temporal_scope` 为 `TemporalScope | None`（P3.1 起）：`None` = 尚未审核（P3 草稿场景的默认值），`TemporalScope.unknown` = 已审核但无法判断。**不要用 `unknown` 掩盖未审核状态。**

枚举层面的 `unknown`（`ContentScope` / `TemporalScope` / `RealityStatus` 同理）表示"经过判断但无法归类"，与"还没判断"（字段缺省、流程未到）不同。

## 4. P2 如何使用 ScriptSegment

`ScriptSegment` 是 P2 解析器（逐行状态机）的输出契约：

1. **每条台词一个 `dialogue` 段**：`speaker` 为说话人标签（如 `妃`、`琉璃`），`quote_style` 按引号类型填 `corner` / `double_corner` / `curly`（**不得为 `none`**，模型校验强制）；
2. **叙述合并为 `narration` 段**：`speaker=None` 且 `quote_style=none`（两者均为模型强制的跨字段约束，状态机分类错误会被直接拒绝）；
3. **行号溯源**：`source` 记录该段的 `source_path + line_start..line_end`（合法跨行台词跨多行；`line_end` 含闭引号所在行）；
4. **告警显式传递**：状态机遇到的异常（如原文未闭合 `「` 的 7 处截断、说话人标签不可靠）写入 `warnings: list[str]`，**不静默丢弃、不静默吞并**；
5. **模型只表达结果**：解析逻辑（逐行状态机、告警判定）在 P2 的解析器模块中实现，`ScriptSegment` 仅承载解析产物，供回归断言使用（P0 审计 §2.2 的 7 处未闭合位置与 9 条合法跨行台词可作为 P2 验收基准）。

### 4.1 `text` 字段口径（P1.1 确立）

`ScriptSegment.text` 的唯一口径：

- **保留原文中的引号字符**（开引号、闭引号、内侧嵌套引号原样保留；原文重复闭引号瑕疵如 `」」` 也原样保留并经 `warnings` 记录）与**跨行文本**；
- **换行契约（P2.1）**：输入的 CRLF/CR/LF 统一规范化为 LF——这是**唯一允许的文本规范化**；`text` 保留物理行边界与行内原文，但不承诺保留源文件的换行编码；
- **不包含行首的 `[说话人]` 标签**——标签解析进 `speaker` 字段，`text` 从开引号起；
- **不做语言清洗、改写或引号修复**：不规范化省略号、不修补标点、不合并分行、不补闭引号；
- **原文未闭合引号时保留实际文本**（含残留的开引号），并通过 `warnings` 记录（如 `unclosed_quote` + 截断位置）；未闭合台词仍是 `dialogue`，不降级为 narration。

示例：

| 原文行 | text | 说明 |
|---|---|---|
| `[妃] 「我讨厌大海，\n受不了海风吹乱头发。」` | `「我讨厌大海，\n受不了海风吹乱头发。」` | 跨行台词保留引号与换行 |
| `[夜子] 「就一点点…时间。`（未闭合） | `「就一点点…时间。` | 保留开引号 + `warnings=["unclosed_quote:..."]` |
| `她的名字叫月社妃。` | `她的名字叫月社妃。` | narration 无引号、无标签 |

## 5. 已知边界（P1 范围外）

- 不含 `embedding_text` / FAISS / BM25 / 数据库主键 / API 响应字段（当时规划由后续索引阶段决定）；
- `SourceSpan` 不校验路径存在性（P1 不读文件）；
- `mentioned_characters` / `present_characters` 的抽取方式（P4 本地 LLM + 人工审核）不在本阶段定义；
- 人物缺席状态不存储，由 `speakers / mentioned_characters / present_characters` 运行时推导（P0.2 决议）。

## 6. P4A 场景元数据审核契约（追加，2026-08-26）

P4A（`backend/knowledge/game_rag/scene_metadata_review.py`，设计文档见 `KISAKI_GAME_RAG_SCENE_METADATA_REVIEW.md`）在 P1 模型之上定义场景元数据审核契约，不改动 P1–P3 既有模型：

- **输入门禁**：`load_frozen_scene_bundle(scenes.jsonl, boundary_manifest.json)` 只接受 `boundary_review_status=approved` 且 `scene_review_status=draft` 的冻结包；两文件缺一不可、场景集合/单元计数/边界/span 有序性/`\x1A` 全部校验后才返回 `FrozenSceneBundle`（携带 manifest/scenes/bundle 三摘要作为稳定关联键）。
- **审核记录 `SceneMetadataDecision`**（`extra="forbid"`）：scene_id / story_unit_id / source 从冻结场景复制后不得更换；viewpoint / temporal_scope / reality_status / mentioned_characters / present_characters / evidence 为待审核字段。
- **viewpoint 规范**：规范化字符串——`<人物名>第一人称`（人物名无白名单，正则约束）/ `第三人称` / `多视角` / `unknown`；`None`=尚未审核。
- **三态语义**（§3 的扩展）：`None`=尚未审核；`unknown`=已审核但无法判断（temporal_scope/reality_status 用枚举，viewpoint 用字符串）；**空人物数组**=已审核且确认无（不得用空数组冒充未审核）。approved 记录不得保留任何 None 待审核字段。
- **顶层审核状态 `SceneMetadataReviewDocument`**（`extra="forbid"`，当前 schema v2）：schema_version / source_manifest（含 manifest_sha256、scenes_sha256、bundle_sha256，绑定冻结双文件内容及组合）/ total_source_scenes / reviewer / review_status（draft|approved）/ scene_decisions（每场景恰好一条）/ notes / created_by；不含时间戳与随机 UUID（确定性创建）。
- **人物集合**：speakers ≠ present ≠ mentioned——书中故事/回忆/梦境/转述可打破"有台词=现实在场"，speakers 仅作人工审核参考，不自动填充。
- **应用流程**：`apply_approved_scene_metadata` 只接受全 approved 状态，写入五个元数据字段并把 `SceneDocument.review_status` 置为 approved；text/id/source/speakers/story_unit_id/顺序守恒。
- **enriched 写出**：`write_enriched_scenes(bundle, review_doc, out_dir)` 只接收与 bundle 摘要绑定的 approved 审核文档，并在内部应用元数据；不接受外部构造的 SceneDocument 列表。

## 7. P4B 场景元数据候选契约（追加，2026-08-26）

P4B（`backend/knowledge/game_rag/scene_metadata_candidate.py`，设计文档见 `KISAKI_GAME_RAG_SCENE_METADATA_CANDIDATE.md`）在 P4A 契约之上定义**模型候选生成**契约，不改动 P1–P4A 既有模型：

- **四类产物不得混淆**：模型候选（`SceneMetadataCandidate`，模型输出）≠ 人工审核决定（P4A `SceneMetadataDecision`）≠ approved enriched 输出（P4A write 流程正式产物，候选路径不可达）≠ 运行 manifest（`CandidateRunManifest`，唯一允许携带时间戳等非确定性信息）。
- **模型客户端协议**：`CandidateModelClient` 为可注入 callable（`prompt: str → str`）；领域模块不初始化服务器、不下载模型、不读取环境密钥；`KeyboardInterrupt` 不被吞掉（断点续跑基础）。
- **候选运行状态 `CandidateRunState`**（`extra="forbid"`，schema v1，确定性创建）：复用 P4A `SourceManifestRef` 绑定冻结包三摘要（manifest_sha256 / scenes_sha256 / bundle_sha256），跨 bundle 恢复或合并被拒绝；摘要校验发生在任何模型调用之前（零调用拒绝）。
- **单场景状态 `SceneCandidateState`**：status（pending/success/failed，运行状态层，与人工 ReviewStatus 语义无关）；success 必携带候选；`last_failure` 只存 error_kind + 异常类型名 + attempts，模型客户端与解析异常消息原文均不落盘（防密钥/URL/模型非法值/回显 prompt 泄漏）。
- **显式重跑审计**：已成功场景不被无意覆盖——重新生成必须显式指定 scene_id + 非空白 rerun 原因，`RerunRecord`（原因/前状态/结果）留痕；重跑失败时旧候选与 success 状态保留。
- **严格解析**：拒绝 markdown 围栏、非对象 JSON、重复 JSON 键、额外字段（`extra="forbid"`）、非法枚举/viewpoint、空白人物名或 warning、scene_id 不匹配、evidence 越界（相对场景 span，分片调用时相对分片 span）；evidence 的 source_path 由解析器从场景补齐。
- **人物数组**：沿用 P4A 规范化（去空白、去重、`sorted(set())` 稳定排序）；空数组=已审核且确认无人。
- **长场景分片**：由 `generation_params.chunk_max_lines`（默认 150）真实控制分片，evidence 与决定**归并回原 scene_id**，不重切 P3 场景；分片分类字段意见不一致 → 取 unknown 并追加告警；单分片重试耗尽 → 整场景 failed，不产出部分归并候选。
- **候选合并 `merge_candidates_into_review`**：人物字段仅 `None` 表示未审核，人工空数组表示确认无人并受保护；warnings 保留人工内容并追加候选告警；值冲突默认跳过并报告（`on_conflict="overwrite"` 为显式字段级覆盖，追加审计告警）；approved / rejected 记录与 reviewer / review_status 绝不被候选改写；合并后记录最多 `needs_review`，顶层保持 draft；schema v1 审核状态拒绝合并，不静默迁移。
- **提示契约 `CategoryHint`**：对梦境/回忆/书中故事/宣传元叙事/魔法重现/无法判断提供结构化类别定义与字段倾向，不预设真实语料结论，最终以人工审核为准。
