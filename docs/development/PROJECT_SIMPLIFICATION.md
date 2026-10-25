# Project simplification audit

Baseline: `8511e95d1a27bfc726a4b82ebeb49ce3fd30cdf9`. Date: 2026-10-25.

This is a bounded, contract-preserving inventory, not a claim that every possible
refactor is complete. Failure must not become fabricated success. Optional modes,
domain abstention, authorization, provenance and cleanup are not redundant merely
because they have fallback-like names.

## Inventory and decisions

| Subsystem | Removed or consolidated | Deliberately retained |
| --- | --- | --- |
| API and generation | Repeated provider arguments; duplicated filtered admin queries; invented authenticated username | Provider/LoRA choice, scoped SQL parameters, branch exclusions, request limits, persistence |
| Inference and bot | Duplicated async-to-sync bridges; repeated vLLM retry accounting; configured tokenizer errors replaced by estimates | Bounded transient retry, actual exception and cancellation, explicit no-tokenizer mode, provider protocols and usage |
| App, middleware, infra, cache | Duplicate JWT decoding, invalid cache-value clamping, unusable HTTP pool fallback | Valid API-key path, nonempty JWT principal, startup ownership, resource limits, all-resource cleanup with propagated errors |
| Database and repositories | Duplicate task decoding and LoRA metadata readers; exception-driven duplicate migration; corrupt JSON replaced by defaults | Savepoints/rollback, legacy schemas, erasure/receipt integrity, strict metadata, unknown rather than fabricated progress |
| Knowledge and character | Broad-reader compatibility branch; duplicate runtime selection; incomplete evidence defaults; eager model loading | Source ownership, conflict retries, required model/index failure, approved evidence, genuine insufficient-evidence abstention |
| Training and evaluation | Repeated cancellation; database-to-memory fallback; stale/failed/blank replies scored as success; invalid Gold/cleaner inputs accepted | Explicit in-memory/mock/ablation modes, task lifecycle, partial-turn receipts, valid refusals, hardware protection |
| Services | Mandatory generation evidence read directly | Completion admission, delivery leases, model-use accounting, subprocess limits, search errors, independent source authority |
| Frontend and gateway | Duplicate authentication HTTP handling and write-only user cache; feedback errors replaced by empty data; unreachable hash dedup | Shared API errors, explicit event identity, delivery acknowledgements and retry receipts |
| Scripts and deployment | Obsolete container-name polling; automatic model-source switching; duplicate launcher settings | Explicit source selection, health-checked Compose startup, bare deployment, canonical launcher |
| Dependencies and files | Orphan UI files/dependencies, unused development CLIs, unreferenced historical micro-reports | Component source, on-demand CLI workflow, licenses, provenance, operational evidence and regression coverage |

Historical Alembic migrations remain deployment contracts. `interfaces` remains
a package boundary. Root scripts, CI, public assets, documentation and archives
were checked for references before deletion. Runtime models, indexes, logs and
user databases are not source-cleanup targets. Game text and training data retain
attribution; absence of application imports is not evidence that a reproducibility
CLI, dataset or offline experiment is disposable.

## Important behavior changes

- vLLM releases active-request accounting before backoff. Exhaustion preserves the
  original error; permanent HTTP/protocol failures do not switch models or invent
  replies. Streaming and cancellation are not replayed.
- Configured training-store read failures propagate. Invalid task configuration
  fails before other fields are written. Explicit `db=None` remains supported.
- Required SQLite migrations roll back and fail startup rather than leaving a
  partially initialized database. Pool shutdown attempts every close and then
  propagates errors. Unexpected authentication faults cannot authorize a request.
- Missing LoRA trainer metadata leaves unknown progress at zero. Malformed metadata
  fails; weights alone do not imply completed training. Final/latest numeric
  checkpoint discovery remains supported.
- Character RAG construction captures configuration without loading a provider.
  The required load boundary initializes provider and service under the existing
  lock and publishes them atomically. Out-of-scope routing and warm/cold timeout
  checks require no model I/O; actual retrieval still requires valid artifacts.
- Rule-memory writes use the repository's existing scoped reader. Conflict retries
  reread authoritative state; they never fall back to scanning all memory.
- Indexed evidence uses its declared schema rather than missing-attribute defaults.
  Explicitly empty valid content is distinct from malformed evidence.
- Evaluation counts only successful, nonblank responses and exposes
  `scored_responses` and `generation_errors`. Failed turns stop the case and retain
  only earlier completed turns. Nonempty refusals/abstentions remain valid outputs.
  Missing/corrupt Gold data and failed safety/persona checks are not empty success.
- Download CLI sources are explicit; removed `--source auto` was not used by callers.
  Compose startup fails when health readiness fails instead of reporting success.

## Why other branches remain

- Retrieval snapshots, generation-bound cache keys, model/vector validation,
  source revisions and complete-query obligations prevent stale or unauthorized
  evidence. Mixed domains cannot borrow authority from one another.
- Approved-evidence fallback preserves already approved evidence when narrowing
  cannot match; it neither invents a quote nor widens approved spans. Missing raw
  source files and invalid ranges still fail.
- Character strategy selection, unresolved-risk handling and evidence abstention
  are policy outcomes, not hidden provider substitution.
- Memory observations, superseded records, pending conflicts, temporal projections
  and protected claims represent distinct history. Confirmed writes remain
  reportable when a later operation fails or is cancelled.
- Background delivery failures keep explicit retry receipts. Optional diagnostic
  logging does not invalidate an otherwise completed review. Dashboard snapshots
  copy cached state and reject invalid configuration.
- Offline probes, replay tools, experiment fixtures and benchmark compatibility
  entrypoints retain distinct schemas. Explicit mock/ablation choices remain;
  they do not activate after real generation fails.
- TRL contract checks, GPTQ's third-party renamed-symbol alias and optional GPU
  telemetry/thermal protection remain. Removing them would require separate
  package or hardware validation. Already-running training logs secondary storage
  failures without hiding the original cancellation/failure.

## Deletion audit

Removed 90 unreferenced historical micro-change reports, verified byte-identical
to the baseline before deletion. Exact paths, rationale and recovery commit are
in [the deletion manifest](../reports/cleanup-20261025.json). Preserved linked
reports, unique limitations, benchmark/model-run evidence, licenses, provenance
and both archives with their checksum manifests.

Also removed these unreferenced UI files and their exclusive dependencies:

- `src/hooks/use-mobile.ts`
- `src/components/ui/popover.tsx`
- `src/components/ui/dropdown-menu.tsx`

Unused `shadcn` and `only-allow` development dependencies were removed, preserving
component source and documented on-demand generation. Compatible Next.js, sharp
and js-yaml updates address the checked dependency advisories.

The initial pass retained all test files. A subsequent input-aware review found
limited exact duplication and fragmented guard contracts; the changes below
supersede that initial assessment. Formatting in the 34-file CI retrieval scope
was verified AST-identical. Separate subsystem audit notes were consolidated here.

## Test integrity and reproducibility

- Routing fixtures explicitly test real local and cloud budget contracts, including
  local `review=None`, rather than ambient machine configuration. Existing source,
  privacy, failure and answer-completeness assertions remain.
- Evidence fixtures use canonical document/candidate schemas. Generic vector tests
  write explicitly synthetic temporary bundles through the real artifact writer
  and loader; all existing vector-error/no-fallback assertions remain unchanged.
  The actual 1,098-document corpus integration assertions were not altered.
- Tests default `HF_HOME` to isolated temporary storage while honoring caller
  overrides. The ASGI export fixture sends a completed request once, then waits,
  rather than starving Starlette with endless immediate request events. Cleanup,
  cancellation, send-failure and admission assertions remain.
- The optional preference stack is already pinned in
  `backend/requirements-preference.txt`. Its datasets 4.1.1 pin conflicts with the
  production datasets 5 range. CI explicitly replaces the exact training-package
  declarations in its existing generated CPU manifest and resolves that manifest
  together with the preference pins. Production requirements remain unchanged.
- PostgreSQL integration uses isolated databases owned by a non-superuser fixture
  role. CI waits for the official PostgreSQL 17 service before provisioning them.

The tracked approved corpus reproduces 1,098 documents and 409 exact original
spans. Generated vectors are intentionally absent from source. Validation uses
only the official public, ungated Apache-2.0-declared encoder
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, pinned at revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Its five required files total
479,724,528 bytes; snapshot metadata and content hashes were verified. No remote
custom code or credentials are used. The existing offline CPU builder generated
all nine genuine partition files in 90 seconds at approximately 1.15 GiB peak RSS.
CI caches the pinned encoder, verifies it and builds the real index before tests.
Weights and indexes remain ignored and are not distributed in this patch.

Final aggregate counts belong to the validation report, not overlapping focused
run totals. Local tiny-model CPU integration is not user-model training; genuine
index rebuilding is not a production promotion or proof of chat-answer quality.
No paid provider validation or production deployment was performed. Browser smoke
still requires a functioning browser sandbox; successful HTTP smoke is not a
browser test. GitHub-hosted workflow execution must be verified after publication.


## Follow-up: test value and shared evaluation probes

This pass starts from PR commit `2e0a212` rather than assuming that earlier
completion notes prove every part of the repository is already minimal.

- Removed four redundant executed cases: one identical task-dependency input,
  one enum-membership assertion already implied by the exact enum list, and two
  erasure-selection input/assertion pairs repeated in the same module. Retained
  the invalid-type, foreign-source, and unauthorized-source security boundaries.
- Consolidated `test_nominal_preference_guard.py`,
  `test_preference_assertion_boundary.py`, `test_polar_question_guard.py`, and
  `test_advisory_preference_guard.py` into `test_user_fact_guard.py` plus
  `test_reply_guard_replay.py`. All 90 executed cases and distinct linguistic
  boundaries remain. The shared generation harness retains all five original
  message/reply pairs and their assertions. Old files are recoverable at the
  starting commit; their deletion is relocation, not loss of coverage.
- The two 455-line document-import evaluation probes differed only in seed-call
  expectation and session identity. They now use `document_import_probe.py`,
  retaining both documented module entrypoints and their distinct defaults.
  Normalized AST comparison confirms that the shared runner preserves each
  original body. Four new contract cases protect stage binding, CLI behavior,
  and invalid-phase rejection before creating output.

A broad static scan did not justify mass deletion: identical assertion bodies
often cover different semantic inputs. Test count is not a quality target.
Baseline collection contained 10,901 executed cases; four duplicate removals
and four new probe contracts leave that total unchanged. Collection equivalence
is not a claim of runtime success; aggregate test results are recorded separately.
The live provider/PostgreSQL evaluator scenarios themselves remain unexecuted.


## Integration review and observation evidence ownership (2026-10-25)

PR #3 was reviewed and merged at `edec840`. Its exact head `34daed4` has verified
CI results: 10,873 backend passes, one applicability skip, eight warnings, and
43 frontend passes. A separate server snapshot passed 145 targeted checks;
17 affected frontend checks passed locally. An extra Windows backend attempt
stalled during collection and was terminated without completed results.

The subsequent memory change keeps full observation sources solely in evidence,
with observation time and uncertainty in display content. Retrieval text and
persisted/admin records are unchanged. Revocation cleanup uses original evidence
rather than the common display label, preserving unrelated history. Production
code is two lines shorter, with no new semantic rule or abstraction.

After integration, 235 related checks and six-file syntax/Ruff checks passed.
Two new DeepSeek Flash requests verified complete synthetic identity, alias,
negative-alias and location evidence through stored recall, actual selection,
reference compilation and readback (3,755 tokens). The fixed sample's reference
shrank from 3,911 to 3,581 characters and selector input from 5,993 to 5,663;
other packet fields and stored memory remained unchanged. An earlier preflight
copied SQLite over an already opened destination and made zero provider calls;
the corrected test copies before importing the database singleton.

The server frontend was rebuilt with locked Next.js 16.4.1 and both services
restarted. A cross-filesystem rename interrupted the initial artifact switch;
explicit cross-filesystem copying completed it. Readiness/login return 200 and
unauthenticated protected access returns 401 with nosniff. Configuration and
other services are unchanged. These checks do not prove browser rendering,
external-platform delivery, sustained load, large-index performance or Windows
process cleanup. Broad model quality and remaining architectural work stay open.


## Validate proposal evidence before admission (2026-10-25)

A synthetic SQLite reproduction showed a valid candidate plus an invented quote
returning `saved` with one persisted fact, silently omitting the invalid candidate.
The existing normalized-source check now precedes business admission and raises
the existing `MemoryResponseError` for `evidence`. The batch fails before fact
writes while retaining the captured original source. Valid low-confidence and
empty outputs remain business outcomes. Production code is two lines shorter;
no semantic keyword rule, additional reviewer, wrapper or repair path was added.

The latest full-file results cover 82 related parser/scheduler caller files:
1,489 passes, plus syntax/Ruff for 18 modified Python files and repository
integrity. Initial test-collection errors were editing mistakes, corrected before
results were counted. Only corrected files were rerun after the first directed
checks; no manual whole-backend repeat was performed.

The prior commit's CI run 40186588600 actually finished with 10,841 passes,
33 failures, one skip and eight warnings; frontend passed and Docker was skipped.
All 33 failures were in seven tests still expecting raw observation sources in
display content. Updated assertions retain full source evidence, temporal and
subject constraints, version chains and actual compiled references. The original
failed run remains evidence, not a claimed green baseline. Tests for authorization
now supply the actual current quotation and still require denial; invented or
copied historical quotes explicitly fail the model protocol instead of looking
like no new information.

One new DeepSeek Flash call (2,720 tokens) saved and recalled two complete
synthetic identity/study facts through actual SQLite and reference compilation.
This checks writer success, not final-answer accuracy. Backend restart readiness
and login are 200; unauthenticated protected access is 401 with nosniff. Frontend
process and configuration are unchanged. Current-source whole-CI results, browser,
external-platform, sustained-load and remaining architecture work stay separate.


## One exact claim/source read contract (2026-10-25)

Initial recall and pre-send revalidation now share `read_claim_sources` in
`character/memory_read_authority.py`. It deduplicates requested pairs, reads at
most 200 per batch, and rejects duplicate or unrequested returned pairs. Initial
recall previously silently overwrote duplicate source responses while revalidation
rejected them. Source availability remains distinct from an adapter failure;
the shared path does not authorize broad source scans. Three repeated batch-read
implementations were consolidated. An unused four-snippet truncation option was
removed; production callers already required complete evidence. Net production
change across three files: minus three lines, no semantic rules or model stage.

All 198 checks in 11 affected files passed, including actual SQLite scopes,
cross-conversation grants, erasure, malformed returned pairs, 201-pair batching
and second-batch failure. Six changed Python files passed syntax/Ruff checks and
repository integrity. With one frozen clock, actual isolated-storage recall
produced identical complete MemoryItem fields and compiled reference text before
and after. A duplicate initial source response now explicitly fails. No paid
model calls or manual full-backend repeat were made; historical native responses
were not counted as new model verification.

Previous exact-head CI 40188703955 at `7eb72a9` completed successfully: 10,879
backend passes, one applicability skip, eight warnings; frontend and both Docker
builds passed. This resolves the preceding 33 old observation-layout assertions,
not a claim about later source. Only backend was restarted; readiness/login are
200 and protected unauthenticated access is 401 with nosniff. Frontend process
and configuration are unchanged. Browser, external-platform and sustained-load
acceptance remain separate work.

## 2026-10-25：生成结果类型与失败归属

GitHub当前没有待合并PR。此前提交的PR #3已审查并合入主分支，来源读取阶段86d6799的[CI 40190362792](https://github.com/despaoy/multi-personal-chat/actions/runs/40190362792)也已全部通过：后端10,883通过、1跳过、8警告，前端与Docker构建通过。

修复未调用主模型的未知结果类型被统一标成rag/abstained并存档的问题。现在仅接受已声明类型，缺失、未知或非法类型在存档前明确返回503，保留原始异常原因；正常无可用原文及显式弃答继续按实际业务结果返回。复用现有用量归集，前面已经发生的复核只记录真实用量，没有发生主生成时不创建虚构调用。没有新模型、语义规则、错误处理框架或重复计量层；功能文件净增14行，用于去除静默默认分支。

八个直接关联文件156项全部通过，覆盖合法结果、错误结果、存档、提供商身份和真实SQLite用量记录；四个改动Python文件语法、Ruff及仓库完整性检查通过。仅隔离合成资料和模拟传输，没有调用付费模型，没有手动重跑完整后端。上一版本CI不能替代本次提交的后续CI，也不等于完成真实浏览器、外部平台和长期负载验收。

## 2026-10-25：上下文主体判断归选择器

完整合成历史明确要求填写本人的问卷，题目中的“你”指填表人；旧主体正则仍把“你喜欢什么？”的用户偏好候选提前删除，真实服务编译结果为空，模型审核根本没有机会处理。现删除memory_read_subject.py及上下文召回的前置主体否决，主体相关性由已有ContextualEvidenceSelector结合原始问题、完整历史和来源证据判断。仓储范围、生命周期、置信度、来源读取及生成前重核继续承担各自职责；无上下文审核的实验读取路径仍保留既有行为。本次生产代码净减58行，没有新增模型、语义规则、分类器或包装层。

十二个关联文件204项通过，涵盖相同问句在不同历史下的选择和最终编译、来源撤回与跨会话授权。上一提交6b3f23f的[CI 40192158108](https://github.com/despaoy/multi-personal-chat/actions/runs/40192158108)实际为后端10,899通过、2失败、1跳过、8警告，前端通过、Docker跳过；两项失败来自旧合成夹具使用未声明或缺失的结果类型。仅补全这两个文件的合法夹具类型，存档、反馈和计量断言保持原样，11项全部通过。本阶段合计14个文件215项通过，六个Python文件语法、Ruff和仓库完整性通过，没有手动重复完整后端检查。

当前DeepSeek Flash完成3次新请求，共3,982 tokens：实际隔离SQLite召回、上下文主体选择、参考编译、来源重核及共享生成链路保留问卷指代、时间条件和否定，普通角色提问排除用户偏好。首次准备因合成私聊会话未通过统一身份规范化而读不到记录，尚未发送模型请求；更正后两次真实分类通过，最终生成又因测试默认top_p与现用提供商契约不符在发送前停止。同一阶段的两个真实分类结果经完整请求一致性核对后明确标记复用，仅补一次最终生成；没有把它们重复算作新的模型验证。生产数据、配置及进程在隔离抽查期间不变；原始请求和模型响应仅保留在服务器隔离记录，不提交。真实抽查范围有限，剩余时间查询规则、大函数职责、浏览器、外部平台和长期负载继续审查。

## 2026-10-25：上下文召回只按实际相关性排序

在41条完整隔离事实中，复合任务需要选一种喜欢的茶并结合现有设备，旧召回却让24个名额全部被偏好类别加分占据，唯一设备未进入候选；单独询问设备可以召回。现上下文路径不调用旧关键词意图与扩展，不把字段意图作为额外排序模态；原始问题和用户历史仍参与检索，词面及真实语义相关性各融合一次，之后由已有选择器判断用途。非上下文对照实验保留既有参数行为，授权、生命周期、证据完整性和生成前重核不变。功能文件净增1行，减少实际执行的语义规则与排序通道。

同一冻结时间、完整SQLite及真实嵌入对比确认：候选上限仍为24，设备由缺失变为在列，另有23项茶偏好；完整41条存储记录和原始问题不变。此固定案例词面比较164次降为41次，不能推为整体延迟降幅。14个关联文件306项全部通过，两个改动Python文件语法、Ruff与仓库完整性检查通过，没有手动重复完整后端。上一版本704b22e的[CI 40195136832](https://github.com/despaoy/multi-personal-chat/actions/runs/40195136832)已全部通过：后端10,903通过、1跳过、8警告，前端与Docker构建成功，确认前一版本的两个旧测试夹具失败已解决。

两次新DeepSeek Flash请求共11,477 tokens，实际贯穿隔离SQLite召回、语义选择、完整参考、来源重核及共享生成请求的事实回读，茶名来自已保存偏好，设备与不存在其他器具的否定均保留。选择器在本例仍保留全部24项，最小证据集合策略继续审查，不能称已验证最少注入。首次测试的前置统计沿用仓储默认30条上限，在发送模型请求前停止；显式读取完整41条后于新隔离目录完成验证。原始请求、响应和运行数据仅保留在服务器，不提交；生产配置和存储不变。本次不代表完整自然语言做法质量、浏览器、外部平台或长期负载验收。

## 2026-10-25：充分证据集合

选择器任务改为充分且不冗余的集合，保持原始证据、逐项协议及实际调用次数。117项定向检查与5次真实模型请求通过，四组选择数2／5／2／0；范围和限制见[当前实现](../CURRENT_STATE.md)。本项结束，继续整体职责与运行验收。
