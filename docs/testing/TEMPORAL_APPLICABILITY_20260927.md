# r117：日期缺失一致性与来源消息边界诊断

## 运行时改动

`project_temporal_record` 原先只处理 validity_authority=unverified。模型不提供日期时，provenance=unspecified，直接绕过时效投影。同一来源/命题/观测时刻因此可能被当成正常事实，也可能被降为原话。现在二者使用同一来源判断；未知schema、其他生产者、既有生命周期不变。不写回记录、没有额外LLM复核、没有改动系统提示词或guard。

新测试在修改前7项失败（`.tmp/mechanism-r117-before.xml`），覆盖当前住址、当前厌恶、未来、过去、条件、期限及“不吃不等于不喜欢”。缺日期、只开始、只结束和倒置猜测区间应得出相同读取视图，并核对实际召回、字段读取与原行不变。

限制必须保留：这是统一已有读取契约，不是通用时间/命题理解。仍依赖较窄的规则完整支持当前事实。旧真实trace回放中，无日期的明确姓名/住址/专业正常读出；“我的专业是物理学，我读大三”仍是observation。当前“不喜欢”的模型摘要少一个“说”也可能不匹配规范模板。不能把这些无谓的不确定性当理想最终设计。

## 本地验证

- 125项时效相关测试通过（`.tmp/mechanism-r117-focused.xml`），随后新增3项未知/其他authority兼容测试。
- 首次广选1339通过、1失败、2568未选中；失败为训练模块 `test_persona_ppo::test_final_test_and_context_budget_rejected`，在加载策略/参考adapter时权重不同。单独运行仍失败（`.tmp/mechanism-r117-training-isolated.xml`），本轮未改训练代码，不宣称全部测试通过。
- 排除整个persona_ppo文件后，最终1342通过、2569未选中、4警告（`.tmp/mechanism-r117-final.xml`）。与聚焦集合重叠，不累计。
- 新来源消息边界诊断器与投影共60项通过（`.tmp/mechanism-r117-diagnostic.xml`），其中8项新诊断器测试；检查不改变system和当前问题、不更改输入、不接受缺失来源包或非冷读请求。Ruff通过，diff检查通过（存在其他文件已有CRLF提示）。

## 服务器真实HTTP

新fixture：`backend/evaluation/fixtures/temporal_applicability_20260927.json`。基线/修改各6场景、6来源+6冷读问题，全部HTTP200。真实Qwen3-8B-Instruct-AWQ无LoRA；实际ASGI→prepare→generation/guard→save→complete→writer，测试鉴权/inline排队、隔离SQLite，问题时清空近期历史。RAG、可选复核和邻近原话扩展关闭，不是生产登录/PG/RAG联合验收。

基线11次新generation+8writer；修改12次新generation+8writer。generation数量包含重试，不能与HTTP数或答案质量混同。新鲜writer输出/观测时刻不同，不能以两组答案差异单独证明投影的因果效果。

| 场景 | 基线 | 修改 |
| --- | --- | --- |
| 当前绍兴住址 | 正确快捷读出 | 正确快捷读出 |
| 过去嘉兴、已离开 | 没有给出新住址，但“还没告诉你”主体不自然 | 直接把用户经历说成“我去年住在嘉兴”，失败 |
| 条件调岗去湖州 | 无据“你还在杭州” | 相同杜撰，失败 |
| 不喜欢茄子 | 答应避开 | 答应避开 |
| 喜欢花生味道、医生让不吃 | 不推荐，但未正确回答喜欢与限制的区别 | 原生成仍混淆，guard回退“记不清”，失败 |
| 天文学、研二 | 要求重复确认 | 这次正确答出，不能当稳定提升 |

逐链路：历史住址、条件住址、花生限制的完整来源均进入最终请求；条件例没有claim，只有原话也仍杜撰杭州。花生例没有claim，原话存在但guard仍按no_match处理具体偏好断言。研二原话和事实副本均在，字段诊断为unverified。不能把这几项概括成检索丢失，也不能仅靠扩大上下文解释。

## 固定请求：消息边界消融

新增 `evaluation/source_role_ablation.py`，在修改组4份冷读失败/不稳定请求上，每份3模式×3seed，共36次新生成，0writer/检索/guard。完全固定system、当前问题和其他记忆视图，不添加期望答案：

1. full：原请求；
2. separate_packet：把完全相同的原话包从当前问题内移到前一条user-role消息；仍保留原来的不可信历史资料标记、source IDs、观测时间、原文，不升级为system；
3. speech_history：将原文放入先前user-role消息。这个实验会失去原话包的结构元数据，并可能暗示检索片段连续，不是拟采用的生产方案。

结果：历史住址full三次中两次冒领用户经历，另一次正确；separate_packet和speech_history各三次都承认不知道新住址。条件住址full三次都杜撰杭州；两种分离模式各三次均保留未确定。复合字段full一次正确、一次无信息、一次带额外确认；两种分离各三次明确读出专业和研二。花生例三种模式均未稳定区分“喜欢味道”和“不能吃”；separate_packet两次出现“那也和我无关”，不能忽略人物体验退化。speech_history有一次建议询问是否可适量食用，也不是理想服从限制。

36次输入1275–1976tokens，全部stop；本组不是8192窗口容量不足或输出截断。

结论：**消息边界是值得测试的机制变量，但还不是通过验收的默认策略。** 这次没有把分离模式写入运行时，也没有删除assistant历史。必须继续独立场景、多来源、时间修订、指代/连续创作依赖、RAG与原话混合、预算边界，以及完整guard链验证。

证据：`.tmp/mechanism-r117-{baseline,invariant}-{traces.jsonl,manifest.json}`、`.tmp/mechanism-r117-source-roles-{results.jsonl,manifest.json}`；旧trace只读影响审计 `.tmp/mechanism-r117-audit.py`。所有远端写入仅 `/home/boot/lhm/multipersonal-runtime/evaluations/`，生产配置/服务/数据库未改。测试进程已结束、SSH已关闭。当前目标未完成。
