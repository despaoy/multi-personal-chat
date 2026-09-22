# 情境感知证据链：实验与使用说明

本轮改造是实验能力，不代表已证明全面优于旧系统。所有新的生成模型调用默认关闭；没有修改真实 `.env`、部署服务或训练现有 LoRA。

## 1. 四项能力分别改变了什么

| 能力 | 新增实现 | 保留的边界 |
| --- | --- | --- |
| 长期记忆 | 全作用域候选召回、最近用户话题参与检索、模型判断当前是否应使用 | 角色/用户/会话隔离、生命周期、证据原文不可改写 |
| 动态上下文 | 人设与价值观参与策略选择，而不只影响措辞 | 固定策略词表、安全、明确任务、用户边界、理解不确定性 |
| RAG | 接入真实 Cross-Encoder；独立 logit 门槛；角色认知可见性接口 | 引用来源、无标注不猜测知情、父场景/时间线/原文同样过滤 |
| LoRA | 证据条件化 SFT / 偏好数据导出及训练入口检查 | 来源隔离、只监督最后回复、必要证据不能被截断 |

“长期记忆全作用域”指仓储已授权的当前用户可见集合，不是扫描其他用户或角色。该实验路径暂时采用全量读后排序；规模增长时应换成作用域内增量索引，而不能直接恢复“只读最近 100 条”。

## 2. 开关与隐私

按模块分别做消融，不要第一次就一起打开：

```dotenv
CONTEXTUAL_MEMORY_SELECTION_ENABLED=true
CONTEXTUAL_MEMORY_SELECTION_TIMEOUT_SECONDS=30
CONTEXTUAL_DECISION_POLICY_ENABLED=true
CHARACTER_RAG_RERANKER_ENABLED=true
RERANKER_MODEL_PATH=/absolute/path/to/verified/reranker/snapshot
CHARACTER_RAG_CROSS_ENCODER_MIN_LOGIT=0.0
```

上面是实验配置示例，不是生产推荐值。第一次真实开发集实验中，直接对长原文做 BGE 精排的 Hit@1 从 0.90625 降到 0.78125，因此角色 RAG 使用独立开关且仍保持关闭，不跟随通用知识库的 `RERANKER_ENABLED` 自动启用。正在比较文档文本表达；不要仅因模型更大就晋升。

`0.0` 是待校准的原始 logit 门槛，不是 0% 或 50% 的事实正确率。`confidence_kind=uncalibrated_sigmoid` 明确表示输出不是校准后的正确概率。不同 `CHARACTER_RAG_RERANK_TEXT_VIEW`（content / summary / embedding_text）的门槛不能视为可互换。

记忆选择和行为策略复用当前配置的基础 vLLM，不加载角色 LoRA，不递归进入聊天生成。若推理地址指向外部服务，开启意味着会向该服务发送受限的对话上下文，应先确认隐私与费用政策。实现中的“本地模型”适配器名称不构成端点必然在本机的保证。

记忆选择异常时不注入候选；策略选择异常时保留旧策略。查看 `PreparedCharacterTurn` 中的 `memory_selection_*`、`contextual_policy_*` 字段和不含原文的日志。超时、格式错误、输入预算超限应与“模型正确拒绝所有记忆”分开统计。

## 3. 真实精排对照

使用独立输出文件，脚本拒绝覆盖已有报告：

```powershell
python scripts/evaluate_multiscale_semantic_rerank.py --model-path ABSOLUTE_MODEL_PATH --output NEW_RAG_REPORT.json
```

该脚本使用项目原有的 68 条开发问题；固定同一知识库、同一召回和同一嵌入，只替换精排。记录每题的检索 ID、原始分数、实际精排方式，以及 Hit@1、Hit@5、MRR、Recall、nDCG。无法在现有索引中解析的标注单独列出，不悄悄删去。真实模型加载失败或意外降级会终止语义实验，不能生成“假模型成绩”。

这些开发题可能已影响旧规则设计，不能作为最终独立测试集。先看逐题变化，再新建按场景/故事隔离的测试集。检索命中也不等于最终角色回答正确，需要另做生成盲审。

## 4. 角色认知边界

调用多粒度检索时可以显式传入 `KnowledgeBoundary(character_id, scene_position, continuity_id)`。`scene_position` 是经维护的剧情顺序整数，不是自动解析的自然语言日期。

知识文档需要人工审核的元数据，例如：

```json
{
  "knowledge_access": {
    "version": 1,
    "continuity_id": "canonical",
    "known_by": {
      "character_id": {"from_scene": 12, "until_scene": null}
    }
  }
}
```

时间窗左闭右开；必须匹配角色和连续剧情。人物出现在文档里不等于人物知道该事实。文档、父场景和原文证据需要分别审核，不能因一个事实可见就自动公开整段场景。

目前已有知识库没有被自动补上这些标注，普通聊天也还没有自动传入该边界。直接给当前未标注索引启用严格边界，会得到空结果，这是预期的保守行为。

## 5. 训练数据路径

新增导出器不会生成或审核答案，只消费已经审核的数据。每条记录采用 `contextual-evidence-v1`，至少包含：

- `id`、`source_group`、`source_ids`、`split`（train / validation / test）；
- `review_status`（只有 approved 可导出）；
- `query`、`answer`、完整的 user/assistant 历史对；
- `evidence`：每项含 id、source_id、kind（memory / knowledge）、content；
- `answerable`、`supporting_evidence_ids`；可选 `rejected_answer`。

支持证据 ID 只留在元数据中，不把正确答案线索标在模型输入中。正例与难负例使用同一情境：语气相近但事实不同、旧记忆与新记忆、角色知道与不知道，才能检验模型真正利用了证据。

```powershell
python scripts/export_evidence_training.py --input REVIEWED.jsonl --system-prompt TRUSTED_PERSONA.txt --output-dir NEW_EXPORT_DIRECTORY
```

输出分别为 `train.sft.jsonl`、`validation.sft.jsonl`、`test.sft.jsonl` 及对应 preference 文件，以及计数/哈希清单。训练只能使用 train；固定验证用 validation；test 留到最终评估。

SFT 配置必须指定独立 `eval_data_path` 并保持 `packing=false`。超过实际 token 预算时训练会明确失败，不能通过截断必要证据让模型学习“没有证据也照样说”。偏好训练也会检查 prompt 和完整序列预算；默认短预算通常不适合长证据，应在显存允许范围内调整，并验证实际 TRL 模板版本。

## 6. 推进次序与验收

1. 冻结旧系统的逐题输出与代码/数据版本。
2. 单独比较语义精排，并分析失败类型。
3. 比较旧记忆、全候选召回、情境选择三个版本；同时报告错注入和漏召回，不能只报其中一个。
4. 比较通用策略与人设条件策略，重点检查最小情境对和用户边界。
5. 准备来源平衡的证据条件化 SFT，再考虑偏好优化；不延长已失败 LoRA 的训练来碰运气。
6. 最后组合模块，做消融和人工盲审；不能把单元测试通过写成模型质量提升。

冻结模型修订、提示词、索引和测试集哈希。人工确认、独立场景评测及真实推理尚未完成前，保持实验开关关闭。

## 7. 本机离线评测

不启动服务器、不读写用户数据库。模型路径必须是已有本地快照；四位 BnB 与线上 AWQ 不同，结果不能自动等同。

```powershell
python scripts/evaluate_contextual_memory.py --mode offline_model --model-path LOCAL_SNAPSHOT --input backend/evaluation/fixtures/contextual_memory_challenge.jsonl --output NEW_MEMORY_REPORT.json
python scripts/evaluate_contextual_policy.py --model-path LOCAL_SNAPSHOT --output NEW_POLICY_REPORT.json
```

记忆脚本加 `--embedding minilm` 可以启用真实本地向量召回；不加时关闭嵌入，以单独观察上下文选择的作用。策略脚本报告人物配对差异和边界约束，**不把人物之间策略不同视为正确率**。离线解码有 token 和逐步时间限额；不是生产异步推理后端，也不承诺能硬中断正在运行的 GPU 算子。

公开 LongMemEval 只读检索入口：

```powershell
python scripts/evaluate_longmemeval_retrieval.py --input LOCAL_LONGMEMEVAL_S_JSON --dense --per-category 2 --output NEW_LONGMEM_REPORT.json
```

按固定哈希每类选样，不按结果挑题；保留日期、说话者、完整分块覆盖。`answer_` 会话 ID、答案和 `has_answer` 不进入检索表示。只测检索，不是完整记忆抽取/写入，也不是最终 QA 分数。负例在 recall 的分母之外；检索到候选本身不能说明最终产生了幻觉。英文公开基准与中文角色项目仍有域差异。

可额外加 `--hierarchical --context-token-budget 2048` 比较“RRF 会话候选→会话内语义轮次排序”和轮流分配会话的对照。需要同时加 `--dense`。除原 top-k 指标，所有组报告同一证据 token 上限的装包召回和实际消耗；以 MiniLM 分词计量、每轮最多取一个完整片段，不是生成模型真实窗口预算，也不保证片段覆盖答案。此功能只在公开基准脚本，不接入生产记忆库。

### 选择协议与解码消融

`--selection-protocol direct` 使用当前直接分类格式；`structured` 输出用途/主体/时间/根据四个布尔判断；`planned` 在不见候选时先理解问题，再选择。后两者仅为离线试验，没有线上开关或稳定胜出证据。

```powershell
python scripts/evaluate_contextual_memory.py --mode offline_model --model-path LOCAL_SNAPSHOT --input backend/evaluation/fixtures/contextual_memory_transfer.jsonl --selection-protocol structured --output NEW_STRUCTURED_REPORT.json
python scripts/evaluate_contextual_memory.py --mode offline_model --model-path LOCAL_SNAPSHOT --input backend/evaluation/fixtures/contextual_memory_challenge.jsonl --decoding sampled --thinking --seed 42 --output NEW_THINKING_REPORT.json
python scripts/compare_memory_reports.py --input backend/evaluation/fixtures/contextual_memory_transfer.jsonl --report DIRECT_REPORT.json --report STRUCTURED_REPORT.json --output NEW_COMPARISON.json
```

比较器核对输入哈希、逐题标准答案与完整覆盖，重新计分，保留旧报告不变。成对正确要求两题都对且不是服务失败；输出变化本身不计成功。`--case-id` 诊断子集中的不完整对会明确标为不计成对分数。

默认仍是非推理贪心；推理模式要求采样。采样种子由初始 seed 与完整格式化提示确定，同题不受顺序影响，但硬件/依赖差异仍可能影响结果。只提取明确 `</think>` 之后的完整最终输出，不保存推理草稿。`--include-model-output` 可能记录输入派生文本，仅用于获准本地诊断。

本机 8GB GPU 同时只运行一个大模型实验。训练还需要激活、梯度和优化器空间，不能由推理加载成功推断训练一定可行。

人设策略脚本可使用 `--state-source rules` 检验实际规则状态，或 `--state-source semantic` 检验现有选择性复核链；`--state-source semantic_all` 是显式的全量非安全状态复核消融。默认仍为 supplied（隔离策略层），生产估计器默认仍为 selective，未新增线上全量开关。全量模式仍不能覆盖、创建或移除安全处理路径。非供给模式中的预期行为只用于评测，不进入状态估计输入。

策略报告新增逐题状态诊断：预期行为是“必要维度”，并非穷尽所有合理行为，因而只报告必要行为覆盖（分数阈值 0.5），不把额外合理行为算成假阳性。状态按问题计数，不因两个角色重复计两次。

仅比较状态而不重复生成角色策略时可用 `python scripts/evaluate_semantic_state.py --model-path LOCAL_SNAPSHOT --thinking --seed 42 --output NEW_STATE_REPORT.json`。默认全量非安全复核，去掉 `--thinking` 即同原非推理贪心配方；`--review-mode selective` 保留选择性入口。`--include-model-output` 仅适合获准本地诊断，记录的是最终输出而非推理草稿。原合成题未人工仲裁，且注入题 greeting 标签存疑，不能把汇总直接当作真实状态准确率。

状态复核 v2 保留当前消息全文（最多 4000 字符）和最近六条有效用户/助手消息全文（合计最多 3600 字符），不再截取前缀。任一预算超出时不调用复核模型，保留规则状态，诊断为 `fallback/input_budget`；这是明确放弃此次语义复核，不代表长文本已被成功理解。可选行为现包含 `gratitude`，安全与风险解除路径仍由原安全门管理，默认选择性触发未改变。与 v1 比较必须区分协议版本，不能混并模型结果。

### 公开中文角色回复对照

```powershell
python scripts/evaluate_public_character_replies.py --cache-dir LOCAL_PINNED_CHARACTEREVAL_CACHE --model-path LOCAL_SNAPSHOT --output-dir NEW_PUBLIC_REVIEW_DIRECTORY
```

固定 CharacterEval 修订 `c3d44a6fc1790cc8c4b2fd7c01f0c72930655e0c`，按角色分组哈希抽取 8 个角色、每角色 2 题；默认 16 题，不按生成效果挑选。完整结构化人物档案进入临时内存注册表，不修改项目角色配置。比较规则与全量非安全语义复核，仍走项目现有上下文、生成与守卫链路。样例 ID、书名标记和评价维度不作为模型输入；原始语料留在外部缓存，报告通过 ID 定位原文。

此试验不是官方 CharacterRM 分数，也不是月社妃原作一致性检验；预训练污染未知。匿名 `blinded.json` 与映射分开，失败及完整回复保留，不采用上游脚本的首行截断。没有人工评审时不得声明角色质量胜出。首次适配审计因错误假设档案为字符串失败，修复为完整 JSON 后保留原来同 16 个 ID，旧审计不覆盖。

需要直接看到人物资料、原对话及匿名回复时：

```powershell
python scripts/render_public_character_review.py --cache-dir LOCAL_PINNED_CHARACTEREVAL_CACHE --blinded COMPLETED_REVIEW_DIRECTORY/blinded.json --output EXTERNAL_REVIEW_DIRECTORY/review.md
```

渲染器只读匿名文件和已校验的原语料，绝不读组别映射；保留空白人工评分栏，不制造评分。含原文的完整审阅文件必须放在项目版本库之外，路径和已存在文件都会检查。原文以不能自行闭合的文本围栏呈现，不把其中标记视为页面结构或工具指令。

### 完整记忆编译与最终回复审阅

启用情境记忆选择后，最终参考区也使用完整 JSON 证据包，保留原始 claim、全部证据、来源及时间元数据。包在 6000 字符总预算内原子加入，放不下就跳过而非截断；最多五条。`used_memory_ids` 表示真正进入最终提示的条目，不保证每条选择结果都能装入预算。数据仍只进入被转义的不可信用户参考区。未启用新选择器的旧路径保持原有 1000 字符紧凑编译行为。这是证据保真修复，不是已证明的回答质量提升。

记忆/情境策略复核的历史窗口仍至多 12 条，但窗口内不再截取单条消息尾部；总计超过 6000 字符就明确回退，避免丢失引用归属或否定。长记忆正文不再因超过 500 字符被置空，改为完整保留并接受 18000 字符总输入预算校验。预算回退不会将扩大召回的候选直接注入回复；生产开关仍默认关闭。

策略层以真正装入的记忆 ID 判断是否有可回忆证据；仅有超预算候选时不能选择共同记忆策略。情境召回与选择共享本轮带时区的接收时间，用于相对时间理解，目前服务默认 UTC，不自动猜用户所在时区。底层接口可显式提供带时区的 `reference_time`；未提供时离线选择器不增添时钟字段，以保持已有冻结提示可重放。

```powershell
python scripts/evaluate_contextual_replies.py --model-path LOCAL_SNAPSHOT --output-dir NEW_REPLY_REVIEW_DIRECTORY
```

使用现有角色上下文服务、生成请求与输出守卫，比较：规则；全量非安全状态复核；全量复核加情境策略。只生成合成题对照，不访问真实数据库、不回写、不加载 LoRA 或 RAG。所有组使用相同非推理贪心配方。环境中若打开情境记忆或策略开关，脚本会拒绝启动，避免基线被污染。

两个回复脚本默认 `--relationship-stage stranger`，可显式选择 acquaintance/familiar/close；报告会断言并记录真实关系字段。首轮曾因夹具误用 `stage` 而实际回退 stranger，不能按早期文字中的 familiar 解读，原报告不覆盖。`--capture-reply-attempts` 用于获准本地诊断，保留初次与重试最终文本、提示指纹、失败阶段及有限的应用契约错误码；不记录任意提供方错误消息或思考草稿。公开脚本还可用 `--case-id` 选择冻结子集中的诊断题，结果标记 diagnostic_subselection，不能冒充全子集结果。

`review.md` / `blinded.json` 为逐题匿名对照，A/B/C 顺序独立打乱；`mapping_do_not_open_before_review.json` 单独保存组别映射。`report.json` 保留状态、守卫回退及模型失败，不伪造评分。必须先完成人工评审再看映射；目前没有任何人工评分。中途停止时 `progress.jsonl` 只是进度，不是完整结果。

训练偏好入口现可读导出的 conversational JSONL。`--max-length` 和 `--max-prompt-length` 显式控制完整序列和 prompt 预算，要求后者小于前者。安装的 TRL 若没有 ORPO，会在加载模型权重前明确失败；不要把当前环境的 DPO 可用误写成 ORPO 也已可用。旧版 TRL 的独立 prompt 截断参数会同步到相同预算，避免校验通过后又被更小默认值截断。
