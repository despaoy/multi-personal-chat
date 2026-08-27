# 月社妃游戏 RAG 场景元数据候选生成（P4B 设计文档）

- 阶段：P4B 场景元数据候选生成与断点续审基础设施
- 日期：2026-08-26
- 代码位置：`backend/knowledge/game_rag/scene_metadata_candidate.py`
- 测试：`backend/tests/test_game_rag_scene_metadata_candidate.py`（102 项；含真实冻结包创建 262 条 pending 状态、零调用零写入测试）
- 上游依据：`KISAKI_GAME_RAG_SCENE_METADATA_REVIEW.md`（P4A，外部审计后 schema v2）
- P4B 实现与外审阶段未运行真实 LLM；P3.6 已正式冻结，但尚未生成真实候选数据或 enriched 正式产物。

---

## 1. P4B 目标

P4A 建立了"冻结场景 → 人工元数据审核 → approved 应用"的契约，但人工从零填写 262 个场景的六个元数据字段成本过高。P4B 在 P4A 契约之上建设**模型候选生成与断点续审**基础设施：

- 通过可注入的模型客户端为每个冻结场景生成**待人工审核的元数据候选**（viewpoint / temporal_scope / reality_status / mentioned_characters / present_characters / evidence / reasons / warnings）；
- 维护可中断、可恢复的候选运行状态（逐场景原子保存进度）；
- 把合法候选**安全合并**进 P4A 审核文档（只填空字段、最多 needs_review）。

LLM 输出只能是候选，**绝不能直接成为 approved 决定**——这是本阶段的安全底线。

## 2. 与 P4A 的边界

| 职责 | 归属 |
|---|---|
| 冻结包输入门禁、审核状态契约、approved 应用、enriched 写出 | P4A（`scene_metadata_review.py`；P4B 外审仅加固共享的模型重校验入口） |
| 模型客户端协议、候选生成、运行状态、断点续跑、候选合并 | P4B（`scene_metadata_candidate.py`，独立模块） |

- P4B **复用** P4A 的 `FrozenSceneBundle` / `SourceManifestRef` / `SceneMetadataDecision` / 原子写协议（`_atomic_write_text` / `_atomic_write_pair`）与规范化函数（`_normalize_viewpoint` / `_normalize_character_names`），不重复实现；
- P4B **不向** `scene_metadata_review.py` 堆积逻辑；
- 候选路径**不可达** approved 应用：模块不导入 `apply_approved_scene_metadata` / `write_enriched_scenes`（架构守卫测试断言）。

## 3. 四类产物（不得混淆）

| 产物 | 载体 | 确定性 | 状态上限 | 说明 |
|---|---|---|---|---|
| 模型候选 | `SceneMetadataCandidate` | 是 | —（非决定） | 模型输出经严格解析后的结构化候选 |
| 人工审核决定 | P4A `SceneMetadataDecision` | 是 | approved/rejected | 人工填写/复核后的记录 |
| approved enriched 输出 | P4A `write_enriched_scenes` 产物 | 是 | approved | 正式产物，候选路径不可达 |
| 运行 manifest | `CandidateRunManifest` | 否（含时间戳） | — | **唯一**允许携带时间戳等非确定性信息的产物 |

候选运行状态（`CandidateRunState`）与 P4A 审核状态保持确定性：无时间戳、无随机数；同一 bundle + 同一模型输出重复运行结果完全一致。需要记录"何时运行"时写入运行 manifest，绝不进入状态文档。

## 4. 模型客户端协议（依赖注入）

```python
@runtime_checkable
class CandidateModelClient(Protocol):
    def __call__(self, prompt: str) -> str: ...
```

- 领域模块**不初始化服务器、不下载模型、不读取环境密钥**：客户端由调用方按协议注入（真实接入时包装 vLLM/Ollama 等现有 Provider，属运行层工作）；
- 客户端以异常表达失败（`TimeoutError` 或其他异常），本模块负责收敛为可重试的失败摘要；
- `KeyboardInterrupt` / `BaseException` **不被吞掉**：进程中断直接传播，未完成场景保持原状态（断点续跑的基础）。

## 5. 候选运行状态

### 5.1 CandidateRunState（顶层文档）

全部 Pydantic v2、`extra="forbid"`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | int | 当前 1（`CANDIDATE_RUN_SCHEMA_VERSION`） |
| `source_manifest` | SourceManifestRef | 复用 P4A 结构：含 manifest_sha256 / scenes_sha256 / bundle_sha256 三摘要 |
| `total_source_scenes` | int（≥1） | 源场景总数 |
| `model_id` | NonEmptyStr | 模型标识字符串（不得含密钥） |
| `generation_params` | dict[str, 标量] | 参数快照（与内置默认合并）；`chunk_max_lines` 必须为 ≥1 的整数并实际控制分片 |
| `scene_states` | list[SceneCandidateState] | 每个输入场景恰好一条（模型强制） |
| `notes` / `created_by` | str | 语义说明 / 生成器标识 |

**三摘要绑定**：跨 bundle 恢复或合并由 `validate_candidate_run` 拒绝（digest 失配、场景集合不一致、scene_id/source/story_unit_id 篡改、schema 版本不符均报错）。**摘要校验发生在任何模型调用之前**（零调用拒绝，测试断言）。

### 5.2 SceneCandidateState（单场景状态）

| 字段 | 说明 |
|---|---|
| `status` | `pending` / `success` / `failed`（运行状态层，与人工 ReviewStatus 语义无关） |
| `candidate` | status=success 时必须非 None；否则必须 None（模型强制） |
| `attempts` | 累计模型调用次数（含分片调用） |
| `last_failure` | 最近一次失败摘要；成功即清空；显式重跑失败时保留（status 保持 success，旧候选不损坏） |
| `rerun_history` | 全部显式重跑审计记录 |

### 5.3 FailureSummary（失败摘要）

| 字段 | 说明 |
|---|---|
| `error_kind` | `markdown_fence` / `invalid_json` / `duplicate_json_key` / `invalid_output` / `schema_violation` / `scene_id_mismatch` / `evidence_out_of_range` / `timeout` / `model_error` |
| `detail` | 截断至 200 字符（`FAILURE_DETAIL_MAX_CHARS`） |
| `attempts` | ≥1 |

**安全设计**：模型客户端异常和候选解析异常的消息原文均**不落盘**。自定义校验消息也可能包含模型提供的非法值，因此持久化 detail 只记录异常类型名，具体失败原因由 `error_kind` 表达。完整 prompt、模型全文、非法字段值和密钥不进入运行状态。

## 6. 场景选择

`select_candidate_scenes(run_state, scene_ids=None, retry_failed=True)`：

- `scene_ids=None`：全部 pending 场景 +（retry_failed=True 时）failed 场景，按冻结顺序返回；
- 失败重试不需要显式原因（不覆盖任何成功结果）；**已成功场景必须显式指定 scene_id + rerun 原因才会重新生成**；
- 指定不存在的 scene id 直接拒绝。

## 7. prompt 契约

`build_candidate_prompt(scene, span=None, chunk_index=None, total_chunks=None, hint_contract=...)`（确定性）：

- 场景信息（scene_id / story_unit_id / story_title / span / speakers，显式声明 speakers ≠ 在场人物）；
- 原文按**绝对行号**逐行展示（`L<n>: <line>`）；
- 输出要求：只输出一个 JSON 对象、禁止 markdown 围栏、逐字段枚举合法值、**evidence 允许范围显式给出**（`必须落在 L<a>-L<b> 内`）；
- viewpoint 规范显式澄清（P4C 真实模型接入时发现）：第一人称叙述必须写明叙述者人物名（如 `琉璃第一人称`），无法判断叙述者是谁填 `unknown`，并提示「以『我』等第一人称口吻叙述的场景不要误标为第三人称」——契约本身不变，只是让模型能正确遵守既有 `<人物名>第一人称` 规范；
- 人物集合提醒补充（同上）：第一人称叙述者（「我」「妾」等）本身属于当前叙事层在场人物，应计入 present_characters；
- 分片 prompt 声明片段序号（`第 i/n 片`），要求仅基于本片段判断；
- text 行数与 span 行数不一致时拒绝（数据异常，不猜测行号映射）。

### 7.1 提示契约（CategoryHint）

对六类叙事现象提供**结构化**提示（`DEFAULT_HINT_CONTRACT`）：梦境 / 回忆 / 书中故事 / 宣传元叙事 / 魔法重现 / 无法判断。每条含 label、description、temporal_scope_hints、reality_status_hints。

- 只描述**类别定义与字段倾向**（如"梦境 → temporal 倾向 hypothetical；reality 倾向 fictional"），**不预设任何真实语料的结论**——最终判断一律以人工审核为准；
- 契约可注入（`hint_contract` 参数），领域模块不绑定具体模型或 prompt 版本。

## 8. 严格 JSON 解析

`parse_scene_candidate(raw, scene, span=None) → SceneMetadataCandidate`，拒绝（`CandidateParseError`，携带 error_kind）：

1. 非字符串输出（`invalid_output`）；
2. markdown 代码围栏（```，`markdown_fence`）；
3. 非法 JSON / 非对象 JSON（`invalid_json`），重复 JSON 键（`duplicate_json_key`）；
4. 额外字段、缺失必填字段、非法枚举、非法 viewpoint、空白人物名或 warning、空 evidence / 空 reasons（`schema_violation`，extra="forbid"）；
5. scene_id 与请求场景不一致（`scene_id_mismatch`）；
6. evidence 越界——相对场景 span（分片调用时相对分片 span）（`evidence_out_of_range`）。

**evidence 的 source_path 由解析器从场景补齐**（模型只输出行号范围），因此"evidence 与场景 source_path 一致"由构造保证；人物数组沿用 P4A 规范化（去空白、去重、`sorted(set())` 稳定排序）；viewpoint 沿用 P4A 规范（`<人物名>第一人称` / `第三人称` / `多视角` / `unknown`）。

## 9. 候选生成与断点续跑

`generate_scene_candidates(bundle, run_state, model_client, *, scene_ids=None, rerun_reasons=None, max_attempts=3, state_path=None, hint_contract=...) → CandidateRunResult`：

**门禁（任何模型调用之前，任一失败抛 ValueError 且零模型调用）**：max_attempts ≥ 1；运行状态结构合法；三摘要绑定与跨字段一致性校验通过；rerun_reasons 合法（只指向已选中的已成功场景、原因非空白）。

**处理规则**：

- 按冻结场景顺序处理选中场景；失败不影响其他场景；
- 已成功场景无显式重跑原因 → 跳过（skipped，**成功结果不被无意覆盖**，零模型调用）；
- 显式重跑：先记录 `RerunRecord(rerun_reason, previous_status, outcome)`；重跑失败时**旧候选与 success 状态保留**（不损坏旧结果），仅更新 last_failure；
- 单场景（分片）最多 max_attempts 次尝试，耗尽置 failed 并保留失败摘要；
- `state_path` 提供时**每个已处理场景之后原子保存一次**（断点续跑：中断场景保持原状态，已完成场景已落盘）。

## 10. 长场景分片归并

超过运行状态 `generation_params.chunk_max_lines` 的场景按连续分片生成候选；默认值为 `CANDIDATE_CHUNK_MAX_LINES`（150 行），自定义值会真实控制 prompt 分片数：

- 分片只影响**候选生成**，evidence 与决定**归并回原 scene_id**，绝不重切 P3 场景（scene_states 仍每个场景一条）；
- 分类字段（viewpoint / temporal_scope / reality_status）：分片意见一致取该值，**不一致取 unknown 并追加告警**（disagreement 交由人工复核，不擅自择优）；
- 人物数组取并集（沿用 P4A 规范化）；evidence 取并集（各分片 evidence 已被限制在分片 span 内）；reasons / warnings 按分片顺序拼接（reasons 去重保持顺序）；
- 单个分片重试耗尽 → 整场景判 failed，**不产出部分归并候选**。

## 11. 候选合并进 P4A 审核文档

`merge_candidates_into_review(bundle, review_doc, run_state, *, on_conflict="skip") → CandidateMergeReport`：

**门禁（合并前完整校验，任一失败抛 ValueError）**：运行状态通过 `validate_candidate_run`（三摘要绑定同一 bundle）；审核文档通过 `validate_scene_metadata_review`（**schema v1 在此被拒绝，不做静默迁移**）；顶层 review_status=approved 的文档拒绝合并（候选路径不得触碰 approved）。

**合并规则**：

- viewpoint/temporal/reality/人物字段只在 `None` 时视为未审核；**人物空数组是人工确认无人，受保护且不得自动填充**；evidence/reasons 默认空数组可由候选补齐；
- warnings 采用保留人工内容并追加候选告警的方式合并，不因候选空告警覆盖或阻塞人工备注；已有人工字段值相同则幂等跳过；
- 值冲突时默认（`on_conflict="skip"`）跳过该场景并报告冲突字段（`skipped_conflict`）；
- `on_conflict="overwrite"` 为**显式**字段级覆盖策略：仅对 draft/needs_review 记录生效，覆盖后追加审计告警（`P4B 候选显式覆盖人工字段（model_id=…）: <字段列表>`）；
- **reviewer / review_status 绝不被候选改写**；approved / rejected 记录一律跳过（`skipped_final`）；
- 合并后的记录最多置为 `needs_review`，**绝不自动 approved**；顶层 review_status 保持 draft（P4A 只允许人工置 approved）；
- 合并产物必须重新通过 P4A 校验（内部不变量，失败即 RuntimeError）；
- 原输入文档不被修改（返回新文档 `CandidateMergeReport.review_doc`）。

## 12. 原子保存与运行 manifest

- `save_candidate_run(path, run_state)`：单文件原子写（复用 P4A `_atomic_write_text`：确定性 tmp → os.replace，失败清理 tmp、旧文件不变）；写出前结构复查，非法状态拒绝落盘；
- `build_candidate_run_manifest(run_state, run_result, started_at, completed_at)`：先重校验状态与结果分组，并要求 `run_result.new_state` 与传入 state 完全一致；随后记录 schema_version / generator / model_id / generation_params / source_bundle 三摘要 / total_scenes / scene_status_counts / run_counts / attempted_scene_ids / 调用方时间戳；
- `write_candidate_run_manifest(path, manifest)`：单文件原子写；
- `save_candidate_run_with_manifest(state_path, manifest_path, ...)`：状态 + manifest **两文件整体原子提交**（复用 P4A `_atomic_write_pair`：primary=状态先备份再替换，secondary=manifest 最后替换作为完成标志；secondary 失败回滚 primary；发现既有 `.tmp.old` 恢复副本时拒绝覆盖）。两文件必须同目录且路径不同。时间戳只写入 manifest，状态文件保持确定性。

## 13. 安全门禁汇总

| 门禁 | 行为 |
|---|---|
| 真实冻结目录（当前工作区状态） | 可加载 262 场景并在内存创建 pending 状态；未调用模型、未写 candidate_run/manifest（测试断言） |
| bundle 摘要不一致 | 任何模型调用**之前**拒绝（零调用） |
| schema v1 P4A 审核状态 | 合并时拒绝，不静默迁移 |
| 候选生成失败 | 不损坏旧状态（旧文件不变、旧候选保留） |
| 候选路径调用 apply/write_enriched | 架构上不可达（不导入；测试守卫） |
| 顶层 approved 审核文档 | 拒绝合并 |
| 异常消息原文 | 不落盘（只记录类型名） |

## 14. 已知限制

- 真实模型客户端适配（vLLM/Ollama Provider 包装、并发、速率限制）属运行层工作，本阶段只定义协议；
- 分片归并的"意见不一致→unknown"是保守策略：人工可改，但候选层不提供加权或多数决（避免伪精确）；
- `generation_params` 只接受标量（int/float/str/bool），复杂参数结构需调用方自行序列化为字符串；
- 运行 manifest 不记录 token 消耗（真实接入后由运行层补充）；
- 运行 manifest 的 `run_counts` / 时间戳只描述**最近一次进程调用**，不代表跨 smoke、run、失败重试的完整时间线；运行层应另行保存本次进程统计，并从候选状态逐场景 `attempts` 推导全生命周期调用次数；
- 候选合并无字段级选择（整体 fill/overwrite/skip），细粒度合并由人工在 P4A 审核状态上直接编辑。

## 15. 公开 API

从 `knowledge.game_rag` 导出稳定入口：

| 入口 | 职责 |
|---|---|
| `create_candidate_run` | 创建确定性初始运行状态 |
| `validate_candidate_run` | 校验（三摘要绑定 + 跨字段一致性） |
| `save_candidate_run` / `load_candidate_run` | 原子保存 / 读取运行状态 |
| `select_candidate_scenes` | 场景选择（pending / 失败重试 / 指定 id） |
| `build_candidate_prompt` | 确定性 prompt 构建（含分片与提示契约） |
| `parse_scene_candidate` / `CandidateParseError` | 严格 JSON 解析 / 可捕获的公开解析异常 |
| `generate_scene_candidates` | 候选生成（重试 / 断点续跑 / 显式重跑审计 / 分片归并） |
| `merge_candidates_into_review` | 候选安全合并进 P4A 审核文档 |
| `build_candidate_run_manifest` / `write_candidate_run_manifest` / `save_candidate_run_with_manifest` | 运行 manifest 构建与原子写出 |
| 模型：`SceneMetadataCandidate` / `CandidateRunState` / `SceneCandidateState` / `FailureSummary` / `RerunRecord` / `CandidateRunResult` / `CandidateMergeReport` / `CandidateRunManifest` / `CategoryHint` / `CandidateModelClient` / `CandidateGenerationStatus` | 数据契约 |
| 常量：`CANDIDATE_RUN_SCHEMA_VERSION` / `DEFAULT_HINT_CONTRACT` | 版本与默认提示契约 |

不导出：私有分片/归并/失败摘要辅助函数。无循环导入。

## 16. 验证记录（2026-08-26）

```
py -3.10 -m ruff check（game_rag 模块 + P4B 测试）→ All checks passed!
py -3.10 -m ruff format --check → 全部已格式化
py -3.10 -m pytest（6 个 game_rag 测试文件）→ 413 passed（311 P1-P4A 回归 + 102 P4B）
```

P4B 测试共 102 项，覆盖创建与三摘要绑定、严格 JSON、候选生成与断点恢复、可配置分片、人工空数组保护、安全合并、manifest 配对、原子故障注入、公开 API 守卫及真实冻结目录只读接入。

实现过程中修正的三处契约偏差（均为对齐既有契约，未放宽任何断言）：

1. prompt 中 evidence 允许范围格式由 `L1-10` 修正为契约的 `L1-L10`（替身客户端与两项 prompt 测试共同钉住该格式）；
2. 模型客户端异常的失败摘要 detail 由"截断的异常消息原文"改为"只记录异常类型名"——截断无法保证密钥/回显 prompt 不进入运行状态，类型名已满足运维定位需求（`model_error` 分类 + attempts 计数保留）；
3. 两处测试夹具/期望修正：`test_select_retry_failed_toggle` 的脚本未给其余场景提供输出（与"失败不影响其他场景"契约冲突，补齐脚本）；分片归并测试的 `present_characters` 期望 `["汀","妃"]` 与同文件解析测试断言的 `sorted(set())` 规范化（妃 U+5983 < 汀 U+6C40 → `["妃","汀"]`）自相矛盾，按后者修正。

P3.6 已正式冻结：两份审核状态 approved，reviewer `project_owner_01`；真实 scenes/manifest 存在并通过三摘要门禁；仍无 enriched、candidate_run 或 run_manifest 正式数据，模型调用为零。

P4C 修订（真实模型接入暴露的 prompt 歧义，见 §7）：smoke test 中 qwen2.5:7b 对第一人称独白场景输出裸 `"第一人称"`（无人物名前缀，schema_violation）或将第一人称叙述误标为 `第三人称`。修正为在 prompt 中显式澄清 viewpoint 规范（契约不变）并补充叙述者在场提醒；P4B 102 项测试全数通过，无断言放宽。

## 17. P4C 外部验收修订（2026-08-26）

真实 P4C 产物经独立加载与确定性重算：冻结包 262 场景，候选状态 259 success / 3 failed / 0 pending，逐场景累计模型调用 441；P4A 审核文档为 259 needs_review / 3 draft / 0 approved，顶层仍为 draft。质量报告与现有候选状态重新生成结果逐字段相等，人工复核 Markdown 逐字节相等并恰好覆盖 262 个场景。137 表示分片分类字段的分歧告警条数，87 表示含至少一条分歧告警的场景数，两者口径不同且计数闭合。

外部验收确认并修复一项人工数据保护缺陷：运行器原 `finalize` 每次都从空白 P4A 文档重建，重复执行可能覆盖已有人工审核进度。现在首次运行才创建文档；已有文档必须先通过 bundle 绑定校验，再以 `on_conflict="skip"` 合并候选，保留所有人工字段与状态；整体 approved 文档拒绝再次 finalize；合并前后 approved 场景集合必须完全一致。

调用统计口径同步澄清：`model_call_stats.json.client_stats` 保留最近一次进程的真实 token/时延统计；新增 `lifetime_state_stats`，从 `candidate_run.json` 的累计 attempts 确定性推导全生命周期调用 441、理论最低调用 410、重试/失败额外调用 31。`run_manifest.json` 仍按 P4B 契约描述最近一次 invocation，不伪造已丢失的早期 token 或时间线数据。
