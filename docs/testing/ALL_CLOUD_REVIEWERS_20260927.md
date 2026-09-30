# r129：内部 LLM 调用盘点、强模型对照与修正

## 调用范围核实

用户指出“内部判断是否仍使用小模型”。核实后：r128 只将回复生成和记忆提取接到 DeepSeek；语义复核没有注入 estimator，证据筛选与行为决策显式关闭。不是暗中使用 8B，但也没有测试这些可选判断。

实际聊天链路中的独立入口：

| 用途 | 原默认入口 | 本次隔离实验 |
|---|---|---|
| 回复生成 | vLLMClient.generate | DeepSeek answer |
| 记忆提取/操作建议 | memory_llm 独立 HTTP completion | DeepSeek writer |
| 情境语义复核 | semantic_review_adapter 的 vLLM factory | DeepSeek semantic_review |
| 记忆证据筛选 | evidence_selector._local_reviewer | DeepSeek memory_selection |
| 人物行为策略 | contextual_policy._reviewer | DeepSeek contextual_policy |

RAG 的向量编码和 cross-encoder 排序不是聊天 LLM，未替换为 DeepSeek。其他独立 API、训练/离线标注入口不在本次 /api/generate 评测覆盖内；没有宣称整个服务器的模型配置都已切换。

## 评测接线

新增 --cloud-reviewers，显式构建三个可选组件并共享云端连接。保留原有触发条件、解析、超时策略，无新增复核环节。每次调用记录用途、模型、结果、耗时；失败和取消也记账。云端实验禁止 VLLMClient.generate 意外调用，即使业务捕获异常，manifest 仍记录并令评测最终失败。密钥不写日志/配置。

配置开启不等于调用发生；not_needed、empty、fallback、applied/selected 分别统计。关闭可选组件与开启组件的两组结果不能冒充仅改变模型的纯因果实验。

## 修改前全云端结果

沿用 r128 的 17 场景、41 轮用例。42 次回答、40 次 writer，另有 41 次策略选择、16 次证据筛选、4 次语义复核，共 143 次 API 请求。无本地生成调用。

按 r128 的核心任务口径，18 个问题仍为 15 通过、3 失败。现居更新被拒、删除未落地却口头称已处理、角色哥哥正确名字被删三个问题仍复现。部分结构化记忆仍为零，不能把原话召回当成写入成功。开启内部大模型判断增加了 61 次调用，并没有解决这些机制错误；不能据此默认开启全部可选判断。

新发现：topic_and_boundary 的一次语义复核 HTTP200，但 finish_reason=length、输出恰好384 tokens，在 conversation_phase 前被截断，触发规则 fallback。耗时4.259秒，非输入窗口溢出。其余3次语义复核成功。本次证明“强模型已调用”不等于“判断已被系统采用”。

## 两项实现修正

1. 默认轻量检查将 UNPROMPTED_CANONICAL_IDENTITY 降为诊断：名字未直接出现在用户问题中，不足以证明身份冒领或主动跑题，不再单凭这一信号重试/删句。strict 保留旧行为；明确安全与用户边界不变。权衡：无关人名扩写也可能放行，诊断不等于事实正确。新增不同姓名、不同关系的迁移测试，以及一次边界重试后诊断不能重新硬拦的测试。
2. 语义判断输出上限384→768 tokens，给完整状态 JSON 留出空间，不增加输入窗口，不强制生成更多，不新增重试、不修改人物提示词。保留5秒超时，继续记录是否真的成功。

## 修正后真实复测

再次执行全部5个RAG/上下文场景、13轮：13次回答、12次writer、13次策略、4次筛选、3次语义复核，共45次API请求，无意外本地生成。

- 哥哥问题完整保留两个人名，词法诊断仍在，但不重试、不删句；13轮均无生成重试。
- 3次语义复核均 stop 且 applied，用时2.19–2.28秒；六个检查问题核心语义通过。
- 新复测该轮只输出232 tokens，历史生成也有变化，因此不能声称已经用同一个长JSON证明768上限的因果收益；原384截断事实与提高上限的实现已确认，需后续固定请求重放/更多表达测试。
- 插画回答仍带反问，角色第三人称/过度表演等自然性问题未全解决；手机品牌的无关检索及外层abstention状态未修复。
- 本轮没有修改记忆更新、删除执行或准入规则。两项高优先级记忆缺陷仍待下一步实现，目标未完成。

## 回归与证据

- 相关广回归421 passed、3662 deselected、4既有警告；.tmp/mechanism-r129-regression-verified.xml。初次受限执行遇SQLite/临时目录权限错误，明确隔离DB后经获准的本地执行通过。不是全项目全量验收。
- 云端适配器3项测试另行通过；119项guard聚焦与82项reviewer聚焦与广集部分重叠，不累计为总通过数。Ruff与定向diff检查通过。
- .tmp/mechanism-r129-{context,memory,corrected}-{traces.jsonl,manifest.json} 保存实际证据。
- 远端输出位于 /home/boot/lhm/multipersonal-runtime/evaluations/r129-all-cloud-context、r129-all-cloud-memory、r129-corrected-cloud-context。
- 仅工作区及远端隔离快照改动；生产未部署。强模型基准验收路径见 STRONG_MODEL_CHAIN_ACCEPTANCE_20260927.md。
