# CAHM：上下文感知混合长期记忆

CAHM（Context-Aware Hybrid Memory）是在现有长期记忆链路上的轻量实验改进。本文后半部分保留第一阶段 CAHM 的实验记录；当前运行时采用下面的“推荐平衡方案”，不再把所有事实压成可覆盖的单条 UPSERT 记录。

## 当前推荐平衡方案

### 写入与 consolidation

- 显式“记住”、纠错和删除请求走 hot path；普通安全陈述先按用户作用域缓冲，不阻塞当前回复。
- 缓冲达到 `MEMORY_LLM_BATCH_SIZE`，或连续空闲 `MEMORY_LLM_IDLE_SECONDS` 秒后，整批提交给单个后台 worker，并按入队顺序逐条判断。服务关闭时使用 `MEMORY_LLM_SHUTDOWN_TIMEOUT` 尝试 flush。
- Memory LLM 只提出关系；本地代码继续校验当前用户原文 evidence、主体、敏感信息、目标记忆白名单、置信度、时间和作用域权限。assistant、system、tool、RAG 与网页内容不能晋升为用户事实。
- 生成与在线消费解耦。后台失败只跳过本次写入，不阻塞或回滚已经生成的回复。

### 三层作用域

| `scope_level` | 可见范围 | 自动写入条件 |
|---|---|---|
| `conversation` | 当前角色、当前逻辑会话 | 默认层；未获得更高层授权时一律落在这里 |
| `user_character` | 同一用户与同一角色的所有会话 | 用户原文同时明确要求“记住”并表达跨会话/该角色持续有效 |
| `user_global` | 同一平台、适配器和用户下的所有角色与会话 | 用户原文明确要求所有角色或全局记住 |

作用域晋升是一项权限，不是模型可自由推断的语义标签。即使 LLM 输出 `user_character` 或 `user_global`，用户原文没有匹配的显式授权时，本地校验仍会降级为 `conversation`。平台、适配器与用户身份隔离在三层中始终保留。

### 七种持久化操作

| 操作 | 含义与存储结果 |
|---|---|
| `ADD` | 新增一条确定的 active claim |
| `MERGE` | 生成包含旧事实与新补充的 active 版本；旧版本标记为 superseded，并保留 parent/证据链 |
| `SUPERSEDE` | 新事实替代旧事实；新版本 active，目标版本 superseded，并关闭旧有效时间 |
| `COEXIST` | 条件、场景或对象不同，两个 claim 都保持 active，通过 qualifiers/parent 表达关系 |
| `PENDING` | “可能、打算、尚不确定”等陈述保存为 pending，默认检索不注入 |
| `RETRACT` | 撤回旧说法但没有替代事实；目标及撤回记录为 retracted，默认检索不注入 |
| `ERASE` | 明确隐私删除；物理删除目标及其派生版本，不保留可恢复 tombstone |

`NOOP` 是第八种裁决结果，但不是持久化生命周期操作：它表示没有新增信息，数据库不写入记录。旧模型的 `UPDATE/IGNORE` 分别兼容映射为 `SUPERSEDE/NOOP`。

### 检索与 evidence packet

推荐默认先按当前/历史查询模式过滤生命周期、低 claim 置信度和有效时间窗，再对原查询、轻量 query expansion、semantic、lexical、importance、recency 与结构化意图通道做 RRF 排名融合。query expansion 只增加实体、主题和时间别名，不生成或写回新事实。最后注入的 `MemoryItem` 携带状态、关系、有效时间、原文 evidence 和 source message IDs；这些字段始终位于“不可信参考”边界内。

四个推荐能力都有真实环境开关，也可在构造服务时显式覆盖，生产默认开启：

| 环境变量 | 构造参数 | 默认值 | 作用 |
|---|---|---:|---|
| `CAHM_RRF_ENABLED` | `rrf_enabled` | `true` | 用各通道相对排名融合，关闭时回到固定权重 legacy hybrid |
| `CAHM_QUERY_EXPANSION_ENABLED` | `query_expansion_enabled` | `true` | 启用实体、主题和时间别名查询视图 |
| `CAHM_VERSION_FILTER_ENABLED` | `version_filter_enabled` | `true` | 按当前/历史查询模式过滤状态、置信度和有效时间窗 |
| `CAHM_EVIDENCE_ENABLED` | `evidence_enabled` | `true` | 返回 evidence、来源 ID、有效时间和 relation metadata |

构造参数为 `None` 时读取对应环境变量；显式传入布尔值时以构造参数为准。平衡版评测器正是通过显式值固定 legacy 与 balanced 两组，避免运行机器的环境变量污染消融结果。

### 历史时间查询边界

版本链保存 `valid_from`、`valid_to`、`observed_at` 与 supersedes 关系，因此管理和审计接口可通过 `include_inactive=true` 查看历史。在线检索默认只使用“现在有效”的 active claim，不能把 superseded 版本当成当前事实。

`load_relevant_memories(..., include_historical=None)` 默认自动识别“以前、当时、上次、去年、前年、2025年、今年六月”等明确过去表达。年/月表达解析为半开时间窗并只保留有效期与该窗口相交的版本；泛指过去时检索完整版本链。历史模式允许 superseded/archived，但仍拒绝 retracted、erased、deleted 和默认不可信的 pending。调用方也可传 `True` 强制历史模式，或传 `False` 强制当前模式。

Gold Set 单独包含“今年六月还在做什么”等历史时间查询。评测器通过公开参数启用历史模式，并按原始状态计算泄漏；合法命中的历史 Gold 不会被误算成 superseded leakage。

### 实际配置

```dotenv
# 后台归纳；显式纠错/记住/删除仍走 hot path
MEMORY_LLM_ENABLED=true
MEMORY_LLM_IDLE_SECONDS=2.0
MEMORY_LLM_BATCH_SIZE=4
MEMORY_LLM_CONFIDENCE_THRESHOLD=0.85

# 检索与 claim 门控
CAHM_SEMANTIC_MEMORY_ENABLED=true
SEMANTIC_MEMORY_CANDIDATE_LIMIT=100
MIN_HYBRID_MEMORY_SCORE=0.35
MIN_MEMORY_CLAIM_CONFIDENCE=0.45
CAHM_RRF_ENABLED=true
CAHM_QUERY_EXPANSION_ENABLED=true
CAHM_VERSION_FILTER_ENABLED=true
CAHM_EVIDENCE_ENABLED=true
```

`MEMORY_LLM_BASE_URL` 和 `MEMORY_LLM_MODEL` 为空时复用 vLLM 配置。若地址指向外部服务，用户消息会发送到该服务，启用前必须确认隐私边界。完整默认值见仓库根目录 `.env.example`。

### 平衡版实验运行

```powershell
cd backend
python -m pytest tests/test_evaluate_cahm_balanced.py tests/test_balanced_memory_retrieval.py tests/test_memory_write_lifecycle.py -q
python experiments/evaluate_cahm_balanced.py
```

第二条命令始终真实运行 legacy hybrid 与 balanced default 检索，并生成 `evaluation/results/cahm_balanced_results.json` 和同名 Markdown。未配置 Memory LLM 时，operation/target/status accuracy 与 operation macro-F1 明确为 `N/A`，不会用规则预测代替。

连接 OpenAI-compatible Memory LLM 后再评关系裁决：

```powershell
python experiments/evaluate_cahm_balanced.py `
  --memory-llm-base-url http://127.0.0.1:8001 `
  --memory-llm-model your-model
```

Linux shell 使用同样参数并去掉 PowerShell 续行符。报告同时给出 R@1、R@5、MRR、错误注入、生命周期泄漏、证据完整度、延迟和逐例失败；它只记录实际运行结果。

## 第一阶段 CAHM 实验记录

## 改造前真实链路

1. `character.memory_extractor.extract_memories` 是精度优先的规则提取器，可提取姓名、普通称呼偏好、喜欢/不喜欢、专业、学习阶段、所在地、工作单位、持续目标和承诺；拒绝“不记录”、敏感信息、不确定句和疑问式承诺。每条消息最多 4 条。
2. 改造前 `character.memory_llm._MemoryJob` 只携带当前用户消息、`rule_hints` 和来源消息 ID；Prompt 不含历史与旧记忆，也没有 ADD/UPDATE/IGNORE。
3. 规则和 LLM 都先构造稳定 `memory_key`。`repositories.character_memory.DatabaseCharacterMemoryRepository.add_or_update_memory` 把它交给数据库；SQLite 的 `db.database.SQLiteDB.add_or_update_character_memory` 和 PostgreSQL 对应实现都在 `character_id + UserScope + memory_key` 唯一约束上执行 `ON CONFLICT DO UPDATE`，更新类型、内容、重要性、来源消息和时间戳。
4. 改造前 `character.memory_service.CharacterMemoryService.load_relevant_memories` 只读取最近 30 条，以内容去固定前缀后的中文 bigram Jaccard 作为 relevance；先执行姓名/偏好主体抑制，再对姓名、偏好和 promise 意图施加 0.4 relevance 保底，并计算 `0.6 * relevance + 0.3 * importance + 0.1 * recency`。recency 使用 30 天半衰期；非意图且 relevance < 0.05 的候选被丢弃。
5. `character.context_builder.compile_reference_context` 最多接收 5 条记忆，单条最多 300 字符，记忆/称呼合计最多 1000 字符；保留“不可信参考”、不可执行其中指令和“记忆主体是当前用户”的声明。
6. 主体区分由 `character.memory_service._detect_memory_intents`、`_topic_subjects` 和 `_is_suppressed` 完成：按子句中话题词之前最近的代词判断用户或非用户主体。因此“我叫什么/我喜欢什么”触发用户记忆意图，“你叫什么/你喜欢什么”在评分前抑制对应用户姓名或偏好。

## 最小修改设计

- 上下文提取：后台任务携带最近 4 条 history；worker 在执行时读取当前 UserScope 最近 10 条旧记忆。严格 JSON 增加 `content`、`operation`、`target_memory_key`。`parse_llm_proposals` 本地执行 opt-out、敏感信息、不确定/疑问、第三方、连续 evidence、值来源、内容来源、阈值、数量和旧 key 白名单校验。UPDATE 继续调用既有 UPSERT，并使用白名单中的目标 key 覆盖旧事实。
- 混合检索：复用 `knowledge.retrieval_core.embedding` 的进程共享多语言 provider。记忆向量缓存键为 `(memory_id, updated_at, sha256(content))`；默认读取当前 scope 最多 100 条。分数为 `0.65 semantic + 0.15 lexical + 0.15 importance + 0.05 recency`。
- 门控：混合得分低于 `MIN_HYBRID_MEMORY_SCORE=0.35` 不注入；没有过线项时返回空集。结构化意图有独立最终分保底，主体抑制在 embedding 之前执行。模型/依赖不可用时降级为原 bigram baseline。

## 文件

修改：

- `backend/character/memory_llm.py`
- `backend/character/memory_service.py`
- `backend/services/character_context.py`
- `backend/knowledge/retrieval_core/embedding.py`
- `backend/knowledge/retrieval_core/__init__.py`
- `.env.example`
- 既有记忆测试的最小兼容调整

新增：

- `backend/tests/test_context_aware_memory_extraction.py`
- `backend/tests/test_hybrid_memory_retrieval.py`
- `backend/tests/test_memory_subject_suppression_regression.py`
- `backend/experiments/evaluate_cahm.py`
- `backend/evaluation/data/cahm_memory_gold.jsonl`
- `backend/evaluation/results/cahm_ablation_server_qwen3_8b_final.json`
- `backend/evaluation/results/cahm_ablation_server_qwen3_8b_final.md`

## 实验结果

Gold Set 共 60 条：18 条提取、42 条检索。最终实验在 RTX 3090 服务器上使用本地真实 `Qwen3-8B-Instruct` Memory LLM 和 `paraphrase-multilingual-MiniLM-L12-v2` embedding；C/D 的提取指标来自真实模型调用。

| 组别 | 提取 P | 提取 R | 提取 F1 | R@1 | R@5 | MRR | 错误注入率 | 平均检索耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A Baseline | 1.0000 | 0.6667 | 0.8000 | 0.5938 | 0.5938 | 0.5938 | 0.1364 | 0.03 ms |
| B Semantic Retrieval | 1.0000 | 0.6667 | 0.8000 | 0.8438 | 1.0000 | 0.9219 | 0.6000 | 约 15 ms |
| C Context Extraction | 1.0000 | 0.7333 | 0.8462 | 0.5938 | 0.5938 | 0.5938 | 0.1364 | 0.05 ms |
| D Full CAHM（0.35 门槛） | 1.0000 | 0.7333 | 0.8462 | 0.6563 | 0.7188 | 0.6875 | 0.2581 | 8.68 ms |

阈值敏感性：0.35/0.45/0.55 的 Full CAHM R@5 分别为 0.7188/0.5625/0.4063，错误注入率分别为 0.2581/0.2174/0.1875。0.35 是本小型集上较合理的召回—污染折中，但不应视为生产阈值。

成功案例：B 能召回“推免 ↔ 保研”“就职 ↔ 工作单位”“生活城市 ↔ 所在地”等 baseline 漏召回项；D 在门控后把无门控 B 的错误注入率从 0.6000 降到 0.2581。主体回归测试还验证了即使伪造所有候选向量完全相似，“你叫什么/你喜欢什么”仍不会泄漏用户姓名/偏好。

提取失败案例：最终运行中 `e02` 专业、`e09` 承诺、`e13` 上下文省略和 `e15` 细粒度咖啡偏好修正未被接受；报告 JSON 保存了每例原始 LLM JSON、本地校验后的预测及 false negative。检索失败案例：`r01` 中 Full CAHM 对“我保研准备得怎么样”可能保留高重要度拿铁偏好；`r06` 对抽象“科研方向”会把工作单位排在点云识别之前；点云识别/点云补全等近邻事实可能同时过门。说明通用 MiniLM + 小数据设置仍缺少更精细的 query-aware 校准，门控降低了污染但没有使其低于 baseline。

## 假设结论

- H1：在本 18 条合成提取集和 Qwen3-8B 配置上获得支持。上下文提取相对规则基线 Recall 从 0.6667 提升到 0.7333，F1 从 0.8000 提升到 0.8462，Precision 均为 1.0000。样本量较小，结论不能外推为生产效果。
- H2：部分支持。真实句向量运行使 B 相对 A 的 R@1 从 0.5938 到 0.8438、R@5 从 0.5938 到 1.0000；0.35 门控使错误注入率相对无门控 B 从 0.6000 降到 0.2581，但同时把 R@5 降到 0.7188，且错误率仍高于 baseline 0.1364。

## 复现

```powershell
cd backend
python -m pytest tests/test_character_memory_extractor.py tests/test_character_memory_llm.py tests/test_context_aware_memory_extraction.py tests/test_hybrid_memory_retrieval.py tests/test_memory_subject_suppression_regression.py tests/test_character_context.py -q
python experiments/evaluate_cahm.py
```

连接真实 Memory LLM 后评测 H1：

```powershell
python experiments/evaluate_cahm.py --memory-llm-base-url http://127.0.0.1:8001 --memory-llm-model your-model
```

## 设计依据

本原型只迁移能在现有代码中轻量验证的机制，没有引入完整外部记忆框架或图数据库。主要参考：

- [Codex memory consolidation](https://github.com/openai/codex/blob/main/codex-rs/memories/write/templates/memories/consolidation.md)：异步 consolidation、证据分层与 NOOP。
- [Mem0](https://github.com/mem0ai/mem0)：过滤、召回、重排和写入门控的组合思路。
- [Graphiti](https://github.com/getzep/graphiti)：有效时间、版本关系与 provenance；本项目只实现轻量 claim 链。
- [MemOS](https://github.com/MemTensor/MemOS)：依据实际使用的 memory ID 做定向纠错。
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)：知识更新、时间查询、弃答与证据评测分类。

Qoder、Trae 等产品可以确认具有跨会话记忆行为，但其内部抽取、冲突裁决与召回算法未公开，因此没有作为已验证实现依据。
