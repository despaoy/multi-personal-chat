# r120：记忆写入故障分层审计

上一轮否定默认来源重试，本轮没有增加在线模型调用或修改提示词。

## 结构化输出能力复验

复用 episode_subject_audit 的两个反约束探针，实际当前服务器 Qwen3-8B-AWQ
两次仍输出正文要求的 `{"unconstrained":true}`，而非 schema 强制的 probe 值。
结果 .tmp/mechanism-r120-capability.json。2 次新真实生成，正常 stop。
驱动按设计拒绝开始 constrained benchmark，REMOTE_EXIT=1 是能力门槛不通过。
与 r84 结果一致，未重启服务或变更引擎。故不能仅添加 response_format 就宣称
writer 已获得结构保证；没有运行一组已知未约束的重复 writer 对照。

## 40 次既有真实 writer 输出的当前准入重放

新增 evaluation/memory_writer_audit.py，保留 recorded payload 的历史、旧记忆白名单、
feedback IDs 和阈值。输出原始候选及当前解析器准入结果，0 新模型、0 数据库写入。
输入是 r117-invariant、r118-warm-baseline、warm-separated、rag-scoped 的全部 writer_calls。
结果 .tmp/mechanism-r120-writer-audit.jsonl：

| 状态 | 次数 | 意义 |
| --- | ---: | --- |
| format_failed | 1 | 取消计划的输出有额外闭括号，整批无法解析 |
| empty_candidates | 24 | 模型返回空列表，不等于这些轮次都应该写入 |
| none_admitted | 8 | 有候选但当前准入为空，包含合理拒绝的 NOOP/问题，不等于 8 次漏写 |
| admitted | 7 | 至少一条提议准入，不等于语义正确或新写入成功 |

这是有选择的近期样本，不是生产错误率统计，也不能用 1/40 证明格式问题无关紧要。
确认的问题包括 preference 非法类型、MERGE/SUPERSEDE 缺少目标、问句试图写事实。
单纯修 JSON 不能修复这些语义/协议错误。

## 更严重的归属污染

r118 两组 friend_owner 来源都是“我姐姐在乐山开咖啡店，我自己在宜宾学木工”。
模型将两者均标为 user/location；当前解析器都准入。原始真实 SQLite traces 进一步
确认两条 active `user_location`，其中一条 content 为“用户姐姐在乐山开咖啡店”，
metadata.attributed_to=user。不是仅离线构造的风险，也不是只有规则测试猜测。

现有第三方排除依赖固定亲属/社会关系词表及有限谓词、字符距离，没有覆盖此例。
文字内容保留“姐姐”不能纠正错误字段语义；当前回合依赖原话答对也不能认证存储正确。
本轮不追加“姐姐”特例，不把全部未识别句子直接丢弃，也不清理任何既有库。

下一轮优先设计“来源保留”与“本人字段准入”分离的主体证据契约：明确本人字段才
进入本人槽位，关系/第三方信息留在有主体的来源或关系通道；模糊情况不能靠模型
自报 attributed_to=user 取得权限。需要覆盖所有格主体、自述中的他人宾语、转述、
并列本人/第三方、指代继承与撤回，避免改成更大词表或更窄白名单造成全局召回退步。
此方向尚未落地或验收，不宣称根本问题已修复。
