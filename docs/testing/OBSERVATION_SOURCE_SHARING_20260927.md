# r124：完整原话共享实验与默认撤回

## 结论

只在来源身份、完整文字、观测时间全部一致时，把重复观察参考共享到原话通道，能够缩短输入，但真实回答没有稳定改善。候选保留为 `attach_sources(..., share_observations=True)` 的显式离线实验，**默认关闭**；生产未部署、未重启、未迁移。目标继续 active，不按使用量停止。

## 机制与边界

- 只处理 active/ADD、非历史、无 qualifiers、单 evidence/单 source ID 的 source_observation；普通个人事实、COEXIST、多来源及不完整覆盖均保持原样。
- 先按同一称呼/完整证据编译参数重建原参考，无法精确重建就不优化，不覆盖额外参考内容。
- 保留 memory_packets 和 used_memory_ids，删除的只是重复模型输入，不删除数据库记录。仅原已选中的项目可重新编译，不能利用腾出的预算偷偷引入未选条目。
- transient source_reference_backup 用于替换/清空原话包或关闭候选时恢复参考，不进入模型输入。
- 发现原 system 约束依赖 reference_context 非空；去重使该约束意外消失。修复为共享已准入记录仍保持原约束，未准入的 source-only 检索依然不启用它。未改提示词文字、未新增模型调用。
- 来源检索本身仍为既有受控开关，评测显式开启，不表示生产默认已开启。

## 真实模型实验

同一服务器 Qwen3-8B-Instruct-AWQ，无 LoRA；ASGI HTTP/真实 chat service/隔离 SQLite；测试身份与 inline admission，不是生产账户鉴权与异步队列压力验收。

| 实验 | 新回答调用 | 新 writer | 录制 writer | 结果 |
| --- | ---: | ---: | ---: | --- |
| r124-source-sharing，三场景六轮 | 6 | 0 | 6 | 妹妹与本人归属答对但额外追问；姐姐公司仍冒领；赠礼偏好正确 |
| r124-rag-sharing，两场景四轮 | 4 | 4 | 0 | 两项混合问题归属可读，但插画答复含不必要澄清、角色外解释及服务邀约 |
| r124-sharing-routed，保持原 system 约束 | 6 | 0 | 6 | 妹妹问题只复述未回答；公司归属不明确；赠礼正确 |

共 16 次新回答、4 次新 writer、12 次录制 writer，16 HTTP 200。三个 manifest 均标记 production_modified=false。冷读无近期历史，原话来源完整送达。相关问题的 raw 回答与最终回复一致，未触发 guard retry/fallback；不能把 guard 通过解释成语义通过。

妹妹场景输入字符数：r122 为 3265，初始共享为 2275（-30.3%），保持原约束后 2520（-22.8%）。这是字符统计，不是 token 或延迟测量。两条事件观察仍在数据库、共享 ID 为 1/2；不属于删除资料节约成本。

RAG 两场景 writer 均返回空 memories，0 结构化 claims；回答依赖已记录原话和实际 RAG。两题分别使用 3/2 个 citation；没有 source_shared_memory_ids，故该实验只能验证混合通道可工作，**不能当作共享去重成功的证据**。原话召回成功不等于稳定事实已提取。

r123 固定请求 source_only 实验只替换参考正文，保留 system 和原参考容器；本轮真实 builder 会改变非空条件与容器。不能把前轮 3/3 直接迁移为真实链路保证。各次新 HTTP 的 source ID、时间也不同，未固定 seed，不能从一条答案推断某段提示词的因果效果；当前证据只支持拒绝默认启用，避免继续依赖样例化格式调整。

## 验证与证据

最终相关文件完整回归 **1440 passed / 3 skipped / 4 既有 warnings**，`.tmp/mechanism-r124-final.xml`。相比原按关键字选择的 847 项，本次完整执行其涉及文件，计数不累加。source-sharing/routing 的中间小集与最终集合重叠。Ruff 与 git diff --check 通过，非全项目全绿声明。

覆盖来源不一致、限定/历史/共存保留、称呼保留、重复 attach、替换/清空恢复、备份不入模、混合事实不丢失、system policy 不随已准入数据换通道而消失、默认不启用候选与关闭后恢复。

本地产物：`.tmp/mechanism-r124-{sharing,rag,routed}-{traces.jsonl,manifest.json}`。
远端实验快照：`/home/boot/lhm/multipersonal-runtime/evaluations/mechanism-r124-source`，保留默认启用候选及 routing 修复时版本，**不是本地最终默认关闭版本**。所有远端写入均位于授权 lhm 内，进程结束、SSH 已关闭。

## 仍需处理

1. 正确原话完整入模后仍出现主体冒领、只复述问题；不应归咎于单纯上下文窗口不足。
2. 多个无关记忆是否错误放宽某个具体偏好断言，仍需沿 guard 的逐项支持链路核验，而非增加通用复核模型。
3. 行为限制与真实喜好仍有混淆；原话观察之外的 goal/promise/other_user_fact 不能宣称全面主体安全。
4. 更强模型隔离对照未获回复，不下载或替换模型。后续可继续检查现有机制，但不应反复用同一小样本微调提示格式。
