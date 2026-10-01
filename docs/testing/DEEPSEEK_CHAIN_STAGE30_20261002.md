# 第三十阶段：生成后的原话保存失败必须有真实反馈

日期：2026-10-02，基线 `dbe1c8996e4ab40191fded025ef8fb4c188be597`。只验证一个新边界：合法完整原话通过身份预留、真实 DeepSeek 已生成回复，但最后数据库 capture 明确失败。修复后只复验该失败及解除故障后的完整原话重试。共三个原生请求、三次实际回答、13 次实际 deepseek-v4-pro 调用（基线 4、修复及重试 9），全部云调用 HTTP 200。没有重复旧容量、身份冲突、删除、冷检索、文档导入、搜索或全套回归。

## 完整输入与真实数据库故障

复用原生全部检查通过的 Stage22 `stage3_stage22_owner_only`，首三条本人确认/撤销、朋友确认的完整原始陈述逐项一致。分别克隆 `stage3_stage30_capture_failure`、`stage3_stage30_capture_failure_fixed`，核对一次性集群 data_directory、复制知识索引逐文件 SHA、父数据库四张表摘要。只在复制库重设合成账户密码，实际 login/me 均 200。

新用例完整转述朋友许澄本人收到北汀松语工坊 2026-10-31 周六木版套色课正式书面预约确认，私人回执 KC-843-N，预约仍有效，尚未参加、没有出发；明确只是说话者转述，不是说话者本人的预约。课程、朋友、回执和账户均为合成资料。新 sourceMessageId，原话超过 150 字，完整进入实际模型输入，不补猜遗漏信息。

仅在当前复制库设置真实 PostgreSQL trigger：只拒绝这个新来源 body 非 NULL 的写入，SQLSTATE 55000，pending 身份摘要预留可通过。登录、消息归档、交互统计和真实模型推理不伪造。仓储观察包装只记录实际 capture 返回或异常，再原样返回/抛出。实际观察到 DBAPIError 和该 trigger 的错误标识。所有清理在 finally 删除 trigger/function，复制库确认剩余 trigger=0；生产从未添加故障。

## 根因与原始失败

基线实际 pro 在完整输入下回复“我记下了”，准确复述朋友、课程、日期、有效性及未参加/未出发。随后真实 capture 失败，complete_turn 已正确记录 source_capture=failed / source_capture_failed、未排语义任务。目标仍 pending、body=NULL、摘要存在、terms=0、links=0、可见原话为空、没有目标 claim。

API `_complete_character_turn` 丢弃这个完成结果，两条生成路径也没有接收回写反馈。因此 API 仍返回模型的“我记下了”，warnings=NULL。不能把模型正确复述、聊天存档存在或身份摘要存在当作完整原话持久化。基线保护检查 15/15、独立保存审计 20/22：两项失败分别是没有明确保存告警和回复中没有保存提示，原始失败保留，不能报告整体全绿。

## 服务器优先修复

唯一业务文件为 `backend/api/generate.py`。完成辅助函数返回独立保存提示：明确 capture failed 显示“这条信息的长期记忆保存失败；本次回复不代表已保存，请稍后重试”；revoked/stale 提示已删除且不会恢复；conflict 提示同一消息已有其他内容。recorded、未执行 capture、不支持的旧范围保持原流程，不把语义队列跳过误报为原话丢失。

完成辅助函数的异常或超时只提示“长期记忆保存状态尚未确认……请稍后检查”。不能把 asyncio.to_thread 等待取消等同于数据库事务回滚，也不把未知状态宣称为已保存或明确失败。外部取消仍传播，未转换为成功反馈。没有增加自动模型重试或自动保存重试。

两条生成消费者接收该提示，合并已有 warnings，并在回复末尾独立添加“保存提示”。只有 warnings 而回复没有提示，会让只读取 reply 的客户端继续看到错误的保存承诺，因此两处都传递。模型内容、已有检索告警、引用、confidence 和回答模式保持；保存提示不进入原话 body、模型提示或事实提取，不把朋友信息变成用户本人事实。

原消息归档仍先于 complete_turn，保存实际模型原文，符合既有“消息归档成功才回写”的约束。**本阶段提示在当前 API 响应可见，尚未持久化到聊天历史；重新读取归档会得到原始模型内容。** 不宣称已经修复历史重载反馈。延迟送达后才回写的集成入口仍是独立边界，未在本轮进行实际送达测试。数据库 schema、仓储事务、权限、身份预留、撤销、source-only、语义容量及模型配置均未修改。

## 原生复验与重试恢复

修复后同样真实 trigger 拒绝新原话 capture，API 200 保留真实 pro 内容，同时 reply 与 warnings 明确显示保存失败。原话保持不可检索，没有绕过故障或伪造成功，也不排语义任务。

解除当前复制库的故障后，仅重试相同完整正文、相同来源 ID。pending 身份允许合法同正文重试；本次真实完成 recorded，body 与完整用例逐字一致、摘要清除、terms=112、links=0，独立读取返回完整原话。首次可信收据时间 `2026-10-01T16:49:41.915370+00:00` 原样保留，不用新时间覆盖；后续真实语义 writer 同钟验证，作业最终 processing/inflight=0，没有强制本人 fact。重试回复不再携带保存失败提示。

旧来源、旧 claims、旧存档和父 fixture 四表摘要不变。独立保存审计 28/28，包含真实故障、完整实际模型输入、原文保留、原始归档一致、pending 不可读、失败不排队、精确重试正文、首次时钟、恢复读取、失败提示清除及主体不混淆。作业 PID 2052704、2054463 均终态。模型调用按真实记录 4+9 计数，不把配置请求数当作实际调用数。

## 定向验证与审计控制

新增 `test_memory_completion_feedback.py` 18 项首轮全部通过：完成状态 failed/revoked/stale/conflict/recorded/unsupported/空；真实等待超时、异常细节不外泄、外部取消；两条生成路径分别验证失败、正常、内部无持久化及消息归档失败不回写。保留已有检索 warnings，不改变输入 metadata。只运行这个新文件，使用独立 SQLite 文件，未重跑 Stage29 或全套。

业务 API、测试和新增工具 Ruff 无新增诊断，测试/工具格式化 AST 等价，没有因格式化重复行为测试。工具的动作函数与实测版本 AST 相同，仅增强只读保存审计。

九项保存结果控制全部通过：真实修复被接受；原始失败、缺 warnings、缺回复提示、错误重试正文、篡改首次时钟、伪造模型回复、错误失败状态、伪造数据库故障都被拒绝。审计只处理已有真实响应、数据库、完成结果和云调用记录，没有新增模型或原生调用。

## 运行服务与证据

服务器修复及隔离验证有效后，备份唯一业务文件新旧版本，只重启后端加载。新组 2055716、应用 2055720，就绪 200；进程仍 deepseek-v4-pro、上下文 65536、语义审核 30 秒、原话检索开关 true。provider 配置、backend/.env、启动脚本和其他服务不变，没有生产 generate 请求、显式生产业务数据写入或 schema 迁移。

基线及复验的原始证据分别在私密 runtime `evaluations/r148pg.s3/stage30-capture-failure/` 和 `stage30-capture-failure-fixed/`。原始日志、定向测试、保存审计、反例、备份在 `backups/backend-chain-20261001/stage30/`，生产加载/回滚记录在 `stage30-deployment/`。凭据和生产资料不进入 Git。

```text
python -m evaluation.native_mixed_context_probe --memory-capture-failure --capture-retry --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage30-capture-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

入口只接受已验证一次性集群；require-success 同时检查保护项和独立失败反馈/重试项，不能只看模型 200。明确保存失败反馈已验证；真实 PG 慢写取消后晚提交仍未实测，本轮只验证 coroutine 超时提示语义。消息归档失败反馈、历史重载保存提示及延迟集成入口仍待新证据；Stage24 公开子任务遗漏继续保留，没有后端丢失证据。不能用这个局部阶段宣称全部长期记忆/RAG/动态上下文准确无误，持续目标保持 active。
