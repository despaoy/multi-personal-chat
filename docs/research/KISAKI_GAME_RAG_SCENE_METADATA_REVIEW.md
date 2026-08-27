# 月社妃游戏 RAG 场景元数据审核（P4A 设计文档）

- 阶段：P4A 场景元数据审核基础设施
- 日期：2026-08-26
- 代码位置：`backend/knowledge/game_rag/scene_metadata_review.py`
- 测试：`backend/tests/test_game_rag_scene_metadata_review.py`（84 项；含真实冻结包 262 场景只读加载测试）
- 上游依据：`KISAKI_GAME_RAG_SCENES.md` 与 P3.6 正式冻结（262 场景 / 197 adds / 24 oversized / 133 残留 low）
- P4A 实现阶段未运行真实 LLM；P3.6 后真实冻结包已可加载，但尚未生成正式元数据审核文件。

---

## 1. P4A 目标

冻结后的场景（P3 产物）只解决了"边界在哪"，没有回答检索必需的语义问题。P4A 为每个冻结场景建立**人工元数据审核**基础设施，使后续 RAG 能够区分：

- 谁是当前叙事视角（viewpoint）；
- 当前文本处于现实、回忆、梦境、假设还是魔法重现（temporal_scope + reality_status）；
- 哪些人物只是被提及（mentioned_characters）；
- 哪些人物实际在场（present_characters）；
- 哪些场景能够作为现实事实证据（由上述字段组合过滤）。

本阶段交付：冻结场景输入门禁、审核状态契约、可恢复的审核状态文件（原子保存/读取）、校验器、人工审核 Markdown 包生成、approved 元数据应用流程、合成测试。**不含**：场景切分、LLM 调用、prompt 管理、embedding、检索、数据库写入、API、事实卡抽取——这些属于 P4B 及以后。

## 2. 与 P3 的边界

| 职责 | 归属 |
|---|---|
| 边界候选检测、人工决定、场景组装、冻结写出（scenes.jsonl + boundary_manifest.json） | P3（`scene_segmenter.py` / `low_review.py`） |
| 消费已冻结场景包，建立/校验/保存/应用**内容元数据**审核 | P4A（`scene_metadata_review.py`） |

- P4A **只读** P3 冻结产物，不回写、不重切、不重审边界；
- P4A 不修改 `scene_segmenter.py`（不向其堆积 P4 逻辑），全部新逻辑在独立模块；
- P3 的 `boundary_review_status=approved` 只表示**边界冻结**；`SceneDocument.review_status` 保持 `draft`（内容审核未做）。P4A 输入门禁恰好消费这个状态组合（见 §3）。

## 3. 冻结输入门禁

稳定入口：`load_frozen_scene_bundle(scene_path, manifest_path) → FrozenSceneBundle`。

校验全部在**创建任何审核状态之前**完成，任一失败抛 `ValueError` 且**零文件写入**：

1. 两文件均存在（不接受孤立 scenes.jsonl，也不接受孤立 manifest）；
2. manifest 是合法 JSON；
3. manifest `schema_version` 受支持（当前 `(1,)`）；
4. `boundary_review_status == "approved"`（边界已冻结）；
5. `scene_review_status == "draft"`（P4A 只消费未做内容审核的场景，防止重复审核或消费已 enriched 产物）；
6. manifest 结构通过 `FrozenBoundaryManifest` 全字段校验（`extra="forbid"`，含 units 摘要）；
7. scenes.jsonl 每个非空行可解析且通过 `SceneDocument` 校验（空行容忍）；
8. scene id 唯一；
9. 场景总数 = `manifest.total_scenes`；
10. 每个 story unit 的场景数与 manifest.units 记录一致；
11. `story_unit_id` 须在 manifest.units 登记；同一单元的场景共享同一 source_path，且位于 manifest.source_prefix 下；
12. SceneDocument.story_title 与 manifest 单元标题一致；
13. 同一单元内 span 按行号有序、不重叠，除首场景外的起点须与 manifest.boundaries 完全一致；
14. 场景 text 不含 `\x1A`（DOS EOF 标记不得进入场景文本）；
15. 输入 SceneDocument 的 `review_status` 均为 draft。

通过后返回 `FrozenSceneBundle(scenes, manifest, manifest_digest, scenes_digest, bundle_digest)`。manifest 与按顺序排列的 SceneDocument 全字段分别计算规范化 JSON sha256，再组合计算 bundle sha256。审核状态绑定整体摘要，正文、speakers、source 或顺序发生变化都会失配，不能用另一份“形状合法”的 scenes.jsonl 与原 manifest 重新配对。由于内部 Pydantic 对象本身可变，创建审核、校验和生成审核包时还会重算摘要，拒绝加载后的内存篡改或手工拼装 bundle。

**当前真实工作区已完成 P3.6 正式冻结**。`load_frozen_scene_bundle` 已只读加载 262 场景并验证三摘要；所有 SceneDocument 的内容审核状态仍为 draft，尚不得进入正式索引。

## 4. `None` 与 `unknown`

延续 P1 Schema §3 的约定，三态不得混用：

| 值 | 语义 | 适用字段 |
|---|---|---|
| `None` | **尚未审核**（初始状态） | viewpoint / temporal_scope / reality_status / mentioned_characters / present_characters / evidence |
| `unknown` | **已审核但无法判断** | temporal_scope / reality_status 用枚举 `unknown`；viewpoint 用字符串 `"unknown"` |
| 空数组 `[]` | **已审核且确认没有** | mentioned_characters / present_characters |

**不要用 `unknown` 掩盖未审核状态，也不要用空数组冒充未审核状态。**

## 5. `None` 与空人物数组

- `mentioned_characters=None`：尚未审核；
- `mentioned_characters=[]`：已审核，确认没有被提及人物；
- `present_characters=None`：尚未审核；
- `present_characters=[]`：已审核，确认当前叙事层无人在场。

approved 记录**不得保留任何 None 待审核字段**（模型校验强制）；因此空数组是 approved 记录里表达"确认无"的唯一合法方式。

## 6. 数据模型

全部 Pydantic v2、`extra="forbid"`，不建立继承体系。

### 6.1 SceneMetadataDecision（单场景审核记录）

| 字段 | 类型 | 约束 |
|---|---|---|
| `scene_id` | NonEmptyStr | 非空；创建后不得更换（校验器对照 bundle） |
| `story_unit_id` | NonEmptyStr | 非空；不得篡改 |
| `source` | SourceSpan | 行号合法；不得篡改 |
| `viewpoint` | `str \| None` | 规范见 §6.2；None=尚未审核 |
| `temporal_scope` | `TemporalScope \| None` | 枚举；None=尚未审核 |
| `reality_status` | `RealityStatus \| None` | 枚举；None=尚未审核 |
| `mentioned_characters` | `list[str] \| None` | 人物名去首尾空白；空串拒绝；去重并稳定排序 |
| `present_characters` | `list[str] \| None` | 同上 |
| `evidence` | `list[SourceSpan] \| None` | 每条必须与场景同 source_path 且行号落在场景范围内 |
| `reasons` | NonEmptyStrList | approved 时必须非空 |
| `warnings` | NonEmptyStrList | 审核告警备注，任意状态可用 |
| `review_status` | ReviewStatus | draft / needs_review / approved / rejected |
| `reviewer` | `str` | draft 阶段允许为空；approved 时必须非空 |

模型级跨字段约束（构造时即拒绝）：
- approved 记录不得保留任何 None 待审核字段（viewpoint/temporal_scope/reality_status/mentioned_characters/present_characters/evidence）；
- approved 记录 evidence 至少一条、reasons 非空、reviewer 非空。

**不含**：embedding_text、向量、BM25 分数、数据库 ID、API 字段、LLM token 消耗、prompt 文本、检索分数。

### 6.2 viewpoint 表达（规范化字符串 + 校验器）

采用**规范化字符串**方案（与现有代码风格一致；未采用 `viewpoint_character + viewpoint_mode` 结构化方案，避免为一个字段引入嵌套模型）：

- `<人物名>第一人称`：如 `琉璃第一人称`、`妃第一人称`、`克丽索贝莉露第一人称`；正则 `^\S+第一人称$`，**人物名不做白名单**（代码不假定只有固定角色名单）；
- 固定值：`第三人称`、`多视角`、`unknown`；
- `None` = 尚未审核。

非法值（如自由文本 `随便写的视角`）在模型校验时直接拒绝。

### 6.3 SceneMetadataReviewDocument（顶层审核状态）

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 当前 2（`SCENE_METADATA_REVIEW_SCHEMA_VERSION`；v2 增加 scenes/bundle 摘要绑定） |
| `source_manifest` | SourceManifestRef | 源冻结包稳定标识：schema_version / boundary_review_status / reviewer / total_scenes / manifest_sha256 / scenes_sha256 / bundle_sha256 |
| `total_source_scenes` | int（≥1） | 源场景总数 |
| `reviewer` | str | 顶层审核人；顶层 approved 时不得为空 |
| `review_status` | `Literal["draft", "approved"]` | 顶层状态 |
| `scene_decisions` | list[SceneMetadataDecision] | 每个输入场景恰好一条 |
| `notes` | str | 默认填写语义说明 |
| `created_by` | NonEmptyStr | 生成器标识（`knowledge.game_rag.scene_metadata_review`） |

模型级约束：`scene_decisions` 数量必须等于 `total_source_scenes`；不得有重复 scene_id。

**确定性**：不含时间戳、不含随机 UUID。同一 bundle 连续创建两次结果完全一致。如需记录生成时间等非确定性信息，应放入独立的运行 manifest（本阶段未提供，属 P4B 接入时的运行层产物）。

## 7. 初始审核状态

`create_scene_metadata_review(bundle, reviewer="")`：

- 不修改输入 bundle 或其 SceneDocument（source 深拷贝）；
- 按冻结场景顺序创建记录，scene_id / source / story_unit_id 原样复制；
- 待审核字段全部 None；reasons / warnings 为空数组；review_status 全部 draft；
- reviewer 允许为空（draft 阶段），记录级 reviewer 继承该值；
- **不自动填充**：不用 speakers 填 present_characters（见 §9），不推断视角/时间状态/现实状态；
- 输出可 JSON roundtrip；重复运行完全一致。

## 8. 人工审核包

`generate_scene_metadata_review_pack(bundle, out_path=None) → str`（Markdown，确定性；提供 out_path 时原子写出）：

- 按 story unit（首次出现顺序）与 source 顺序展示全部冻结场景；
- 每个 scene 一节：scene id、source span、story title、speakers、text 行数、填写位（viewpoint / temporal_scope / reality_status / mentioned_characters / present_characters / evidence / reason / review_status，均带可选值提示）；
- **短场景（≤40 行）完整展示原文**；
- **长场景（>40 行）只展示首尾各 8 行摘录**，并显式声明"以上为摘录，摘录不构成完整人工审核，完整原文见 source span L…-L…"——审核长场景必须对照完整 span 原文；
- 包头部说明填写语义（None/unknown/空数组）与"speakers 仅为人工参考"提示。

## 9. 人物集合语义

`speakers`、`mentioned_characters`、`present_characters` 三者不同：

- speakers：原文中有说话人标签的人物（P2 解析结果）；
- mentioned：文本谈到的人物；
- present：**当前叙事层**实际在场的人物。

**禁止自动断言**：speakers 一定等于 present；mentioned 一定包含 speakers；所有有台词者都属于现实层在场人物——书中故事、回忆、梦境、转述和引用台词都可能打破这些假设。P4A 只把 speakers 作为人工审核参考展示在审核包里，不自动填充任何 approved 结果。

## 10. 校验规则

`validate_scene_metadata_review(review_doc, bundle, require_complete=False) → list[str]`（空列表=通过；接受模型或 dict）：

1. 结构校验：dict 须能通过 SceneMetadataReviewDocument 校验（失败返回"结构非法"摘要）；
2. `schema_version` 必须为当前版本；
3. `source_manifest` 与 bundle 逐字段一致：manifest_sha256 / scenes_sha256 / bundle_sha256（钉住双文件内容及组合）、schema_version、boundary_review_status、reviewer、total_scenes；
4. `total_source_scenes` 与 bundle 场景数一致；
5. scene 集合完整：未知 scene 拒绝、缺失 scene 拒绝（重复由文档模型自身拒绝）；
6. scene_id / source / story_unit_id 不可篡改（逐条对照 bundle）；
7. 枚举合法、人物数组合法（去空白/空串拒绝/去重排序）、evidence 落在场景范围内、reasons 结构合法、reviewer 语义合法——由记录模型保证，结构非法即整体拒绝；
8. **顶层 approved 时**：reviewer 不得为空；不得存在 draft / needs_review / rejected 场景；
9. `require_complete=True` 时：全部场景 review_status 必须明确（不得为 draft；needs_review / rejected 属于"已明确审核"）。

校验器不止验证 JSON 结构——3/4/5/6/8 都是跨字段一致性检查（对照 bundle）。

## 11. approved 应用流程

`apply_approved_scene_metadata(bundle, review_doc) → list[SceneDocument]`：

- 先执行完整校验（§10），错误即拒绝；
- 顶层 review_status 必须为 approved：draft 拒绝、needs_review 场景拒绝、rejected 场景拒绝、reviewer 为空拒绝、approved 中存在 None 拒绝（均由校验/模型层保证）；
- 返回**新的** SceneDocument 列表（model_copy，输入 bundle 与审核状态均不被修改）；
- 写入：`story.viewpoint` / `story.temporal_scope` / `reality_status` / `mentioned_characters` / `present_characters`；`review_status` 变为 approved；
- 守恒：text、id、source、speakers、story_unit_id、顺序全部不变；
- 重复运行结果完全一致；
- **不支持局部预览**（不做 `preview=True`），保持流程简单——预览需求由审核包（§8）承担。

## 12. 原子保存协议

沿用项目已验证的原子写模式，不另造事务框架：

`save_scene_metadata_review(path, review_doc)`：

1. 先做结构校验，非法状态拒绝写出（防呆：不会把坏状态落盘）；
2. 内存序列化（`json.dumps`，indent=2，ensure_ascii=False）；
3. 写同目录**确定性**临时文件 `<name>.tmp`（不用随机路径，测试稳定）；
4. 写入成功后 `os.replace` 原子替换；
5. tmp 写入失败：旧文件不变，清理未完成 tmp；
6. 替换失败：旧文件不变（tmp 尚未生效），清理 tmp；
7. 成功后无 tmp 残留。

`load_scene_metadata_review(path)`：JSON → SceneMetadataReviewDocument（结构非法即报错）。save/load 满足 JSON roundtrip 一致。

enriched 输出（`write_enriched_scenes`）使用两文件整体提交协议（沿用 P3 冻结对协议）：两份内容全部写 tmp → primary（enriched_scenes.jsonl）先备份旧版再替换 → secondary（enriched_manifest.json）最后替换作为完成标志 → secondary 失败时回滚 primary，不留混合版本；回滚失败时保留备份并在错误信息中报告路径。备份阶段失败会清理两份 tmp；发现既有 `.tmp.old` 时拒绝覆盖该恢复副本。

## 13. 输出门禁（enriched scenes）

`write_enriched_scenes(bundle, review_doc, out_dir) → dict`：

- 只接受与 bundle 双摘要绑定且全部 approved 的审核文档；函数内部调用 `apply_approved_scene_metadata` 生成场景，不接受外部构造的 approved SceneDocument 列表；
- 独立输出目录（`enriched_scenes.jsonl` + `enriched_manifest.json`），**不覆盖 P3 原始冻结产物**；
- manifest 记录源冻结包稳定标识（含 manifest/scenes/bundle 三个 sha256）与 `scene_review_status="approved"`；
- 两文件原子提交（§12）；
- 本阶段只用合成测试调用，不对真实目录写出任何 P4 数据。

## 14. 失败语义

| 场景 | 行为 |
|---|---|
| 冻结包不完整/不合法 | `ValueError`，零文件写入 |
| 真实冻结目录（当前状态） | 可只读加载并在内存创建 draft 审核状态；不会自动写出 P4 文件 |
| 审核状态结构非法 | 校验返回错误列表 / save 拒绝写出 |
| 审核状态与 bundle 不匹配（digest/篡改/集合不一致） | 校验返回错误列表；应用流程拒绝 |
| draft/needs_review/rejected 参与应用 | `ValueError` 拒绝 |
| 原子写中途失败 | 旧文件不变；tmp 清理；无混合版本 |

所有拒绝都是**显式失败**，不静默降级、不部分写入。

## 15. P4B 如何接入本地 LLM 候选

P4A 的契约即为 P4B 的接口：

1. P3 正式冻结后，`load_frozen_scene_bundle` 加载真实冻结包；
2. `create_scene_metadata_review` 创建初始状态（全部 None）；
3. `generate_scene_metadata_review_pack` 生成人工审核材料；
4. P4B 本地 LLM 按审核包为每个场景生成**候选**元数据（viewpoint / temporal_scope / reality_status / mentioned / present / evidence / reasons），填入 `SceneMetadataDecision`（候选默认 review_status=draft 或 needs_review，**不得直接 approved**）；
5. 人工复核候选：改错、定 unknown/空数组、填 reviewer、置 approved；

## 16. P4D/P4E 真实应用记录（2026-08-26）

P4D 已完成 262 个冻结场景的人工元数据定稿，全部记录先保持 `needs_review`、顶层保持 `draft`。P4E 经项目负责人明确授权后，由 `backend/scripts/approve_scene_metadata.py` 将完整审核文档统一置为 approved，并通过 `write_enriched_scenes` 写入独立的 `scene_metadata_enriched/` 目录。该入口可幂等重跑，不修改 P3 冻结产物，也不进入事实卡、embedding 或索引阶段。
6. `validate_scene_metadata_review`（可加 `require_complete=True`）校验；
7. `save_scene_metadata_review` 原子保存审核状态（可断点续审：load → 修改 → save）；
8. 全部 approved 后 `apply_approved_scene_metadata` 应用，`write_enriched_scenes` 写出 enriched 产物供后续索引使用。

P4B 的 LLM 候选生成、prompt 管理、运行 manifest（时间/模型版本/token 消耗）都在 P4A 契约**之上**叠加，不修改本模块的审核状态结构。

## 16. 为什么本阶段不运行真实数据

- P3.6 已正式冻结；真实 P4 元数据仍须经模型候选与人工审核，不能因为边界 approved 而直接进入索引；
- 真实元数据审核依赖人工判断（视角/现实层/在场人物），LLM 只能出候选，本阶段先把**契约与流程**用合成数据钉死，避免契约返工污染真实审核；
- 禁止事项：不调用真实 LLM、不生成正式事实卡/关系卡/事件卡、不把 P3 状态改为 approved、不调用 `freeze_reviewed_scenes`。

## 17. 已知限制

- 不支持局部预览应用（preview）——审核包已承担预览职责；
- 审核状态无断点续审的增量 API（load → 修改 → save 即可续审，但无字段级 diff 工具）；
- viewpoint 人物名不做白名单：错别字人物名（如 `琉璃第一人成`）会被格式校验拒绝，但同义异写（如 `克丽索贝莉露` vs `克里索贝莉露`）无法自动发现，依赖人工一致性；后续若建人物别名表可加交叉校验；
- evidence 只约束行号范围落在场景内，不校验 evidence 文本内容（审核状态不复制原文，防数据膨胀）；
- enriched 输出的下游消费者（索引/检索）尚未定义（P5+）；
- `warnings` 字段目前无生成方（预留给 P4B LLM 候选的告警传递）。

## 18. 公开 API

从 `knowledge.game_rag` 导出稳定入口：

| 入口 | 职责 |
|---|---|
| `load_frozen_scene_bundle` | 冻结场景包输入门禁 |
| `create_scene_metadata_review` | 创建初始审核状态 |
| `validate_scene_metadata_review` | 校验审核状态 |
| `save_scene_metadata_review` / `load_scene_metadata_review` | 原子保存 / 读取审核状态 |
| `generate_scene_metadata_review_pack` | 生成人工审核 Markdown 包 |
| `apply_approved_scene_metadata` | 应用 approved 元数据 |
| `write_enriched_scenes` | enriched 场景输出门禁 |
| 模型：`FrozenSceneBundle` / `FrozenBoundaryManifest` / `SceneMetadataDecision` / `SceneMetadataReviewDocument` / `SourceManifestRef` | 数据契约 |

不导出：私有原子写辅助（`_atomic_write_text` / `_atomic_write_pair`）、内部规范化函数（`_normalize_viewpoint` / `_normalize_character_names`）、尚未实现的 LLM 接口。无循环导入。

## 19. 验证记录（2026-08-26）

```
py -3.10 -m ruff check  → All checks passed!
py -3.10 -m ruff format --check → 12 files already formatted
py -3.10 -m pytest（5 个 game_rag 测试文件）→ 311 passed（227 P1-P3 回归 + 84 P4A）
```

P3.6：两份边界审核状态 approved；reviewer `project_owner_01`；正式 `scenes.jsonl` / `boundary_manifest.json` 已生成并通过三摘要门禁；SceneDocument 内容审核状态仍为 draft；无正式 P4 元数据或 enriched 产物。
