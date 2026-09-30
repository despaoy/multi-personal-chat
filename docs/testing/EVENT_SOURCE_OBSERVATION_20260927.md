# r122：事件线索与事件参与者分离

## 机制变更

上一轮证明 generic shared_event 可以绕开个人字段保护，并将妹妹的行为写成用户行为。
本轮不扩大亲属词表，也不拒绝所有涉及第三人的事件：

- generic shared_event 的模型输出仅用于事件线索选择，不再采纳其自由改写的 content。
- 保存完整 current source utterance 作为 evidence，包含模型短引文之外的主体和后置限定。
- 短 content 使用带引号的原话记录；长原话的 content 只写“完整内容见证据”，不截断引用。
- metadata 明确 content_semantics=quoted_source、speaker_role=user、described_subject=not_resolved。
  attributed_to=user 在这些记录中表示实际发言者，不认证事件参与者。
- 事件中的他人主体/模型 actor 标签不再作为来源归属；来源必须仍来自实际 user 通道。
  其他个人字段、敏感/不保存/证据/目标/权限检查保留，RETRACT/ERASE 的权限未放宽。
- 事件 MERGE 改为关联 COEXIST 观察：新原话不是累积全文，不能用它 supersede 并丢掉旧细节。
  明确 SUPERSEDE、RETRACT 等关系保持原流程。个人事实 MERGE 行为不变。

这是 generic shared_event 的原话观察表示，不是通用事件角色抽取器。
goal、promise、other_user_fact 等其他类别未因此全部获得主体语义保证。
既有存储不迁移；新证据表示会增加原话长度，重复事件线索的来源内容去重仍可优化。

## 测试

- 广回归 821 passed、3198 deselected、4 既有警告，包含事件观察和冻结writer工具测试：
  .tmp/mechanism-r122-final.xml。
- 随后新增真实SQLite持久化检查；事件组最终13 passed：.tmp/mechanism-r122-events.xml。
  与广集重叠12项，不将两数相加。确认MERGE转COEXIST后旧/新均active、parent正确、
  不设置supersedes、完整evidence和主体未解析元数据持久化。
- 新SQLite测试首跑因默认embedding冷加载超过3秒失败；测试改为注入固定embedding，
  保留真实SQLite与scheduler。该测试不宣称真实embedding性能通过。
- 覆盖本人、亲属、同事、第三人、共同事件、短引文丢主语、后置否定、长文、PENDING、
  qualifiers、非事件不变与source speaker不等于actor。Ruff/diff通过。

## 受控真实HTTP重放

新增 FrozenMemoryWriter，只重放同一 source 的独立ADD/NOOP，拒绝带target的操作，
防止把旧库ID带进新库。原始writer输出保持不变，照常走当前scheduler/解析/真实SQLite。
模型回答是新鲜服务器Qwen3-8B-AWQ推理，而不是重放答案。

从r121修改组选择妹妹/本人、姐姐公司、赠礼偏好三场景，共6 HTTP200、6新generation、
6录制writer输出、0新writer。父亲/共同住址问题的记录含目标依赖，未纳入这个冻结工具；
没有删除它们的真实失败证据，也不声称本轮重新完整测试五场景。
RAG关闭、无LoRA、隔离SQLite、测试身份/inline排队、source recall显式开启、问题冷读。

实际存储：原先“用户在泉州学陶艺”已被包含妹妹与本人的完整原话替代，两个事件线索
都标described_subject=not_resolved。来源没有丢失，妹妹/用户冷读仍正确。
该问题实际输入3221→3265字符（+44），只是该例测量，不是通用成本保证。
姐姐公司场景仍错答“在姐姐公司工作的是我”，完整原话已入模型，且这轮本就没有事件claim；
这是独立的回答阶段归属错误，不能用存储改进掩盖。赠礼偏好仍正确。

证据：.tmp/mechanism-r122-event-{traces.jsonl,manifest.json}；
.tmp/mechanism-r122-writer-audit.jsonl（10份历史输出的当前准入审计，无新模型/写库）。
远端快照mechanism-r122-source，评测进程结束、SSH关闭，生产未改。

下一轮继续追踪回答阶段的来源主体，并检查观察表示的跨轮更正、删除与预算去重；
必须同时保留事实可用性、人物自然度和效率，目标仍active。
