# r136：删除执行前置与真实即时回复

## 实现

增加显式prepare_interactive_turn；普通prepare_turn不引入变更。API仅在持久化聊天且非delivery_context路径启用，prepared_override分支不走此入口。删除意图先验证scope/profile，使用原writer的逐任务有界回执，再读取删除后的上下文。普通事实继续异步处理。complete_turn检测已执行的操作回执，包含pending/failed也不再次提交。关系计数仍走既有完成逻辑。

操作回执按固定枚举转成运行状态数据，附在动态上下文，不携带错误原文或私人事实。不改人物prompt。限定范围是长期记忆及关联来源，未删除聊天历史。

## 本地证据

1594通过、1跳过、2563未选、4既有警告，92.21秒（.tmp/r136-wide.xml）。其后新增内部推理/发送确认模式边界，30项聚焦测试通过（.tmp/r136-boundary.xml），与广回归重叠不累计。Ruff/diff通过。初次导入测试未设置隔离DB导致无法打开数据库；隔离重跑后另有2个测试替身签名未同步，已增加参数后通过，未修改其既有断言。

## 真实强模型结果：尚未通过

服务器独立r136快照，9轮HTTP200，无LoRA、无RAG，冷读追问、来源召回开启。29次新DeepSeek调用：answer9、writer9、policy9、selection2，semantic未触发，0本地生成。来源及数据库均隔离，生产未改。

- 猫名删除：claim/source清空；即时回复“好，已经删掉了。之后不会再引用那条原话。”，后续无名字召回。
- 住址删除：claim/source清空，writer回执erased且生成模型system实际收到确认删除；**模型仍答“我没有保存过你的住址，所以没有需要删除的内容。”** 操作成功但表达失败，不报告全过。
- 否定删除专业：未进入显式执行路径，园艺专业保留且追问答对。
- 冷读追问中的“没有保存过”和邀请重新提供，也不能当成理想删除体验；暖历史残留未在本轮验证。

证据：.tmp/r136-erasure-manifest.json、.tmp/r136-erasure-traces.jsonl。该轮prepared字段捕获的是内部只读prepare结果，因此未包含后附回执；实际入模model_calls/request/messages和completion包含回执证据。已在本地补上prepare_interactive_turn观测，尚未上传/另跑，不能倒填历史trace。

## 下一步

单靠回执入模不能保证操作事实正确。纯操作请求应由已验证回执直接生成确认，不让模型重新猜测；混合请求需保留非操作任务，不能直接吞掉其他问题。还需测试执行后上下文准备失败、客户端取消、pending后完成、重试幂等与暖历史。delivery确认路径当前仍为原有延迟操作，不宣称覆盖。模型能力不是这些链路缺口的充分解释。SSH已退出，目标active。
