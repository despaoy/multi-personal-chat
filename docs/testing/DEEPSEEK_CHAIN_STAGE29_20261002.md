# 第二十九阶段：语义队列满载时仍保存完整原话

日期：2026-10-02（基线验证始于前一日晚间）。基线 `c67f13bacafc3f66410b9d94954d0942d5686c2d`。只新增一个真实语义任务占满容量后保存另一条完整消息的边界，修复后仅复验该失败，并新增一次原话冷检索。共五个原生请求和五次回答，实际 deepseek-v4-pro 调用 23 次：基线 9、修复 14，均 HTTP 200。保存审计和反例控制没有新增模型或原生调用；没有重播旧写入、旧删除、旧问答、搜索、文档导入或全套回归。

## 输入完整、容量真实占用

复用原生全部检查通过的 Stage22 `stage3_stage22_owner_only`，首三个原始本人确认/撤销、朋友确认的完整陈述逐项一致。分别克隆两个一次性数据库，核验 data_directory、知识索引逐文件 SHA、父数据库四张表摘要；只在复制库重设合成账户密码，走正常 login/me。

新占位消息完整转述唐澈本人收到西岭榆湾画室 2026-10-24 周六植物拓印课正式书面确认，私人回执 WQ-318-B，有效、未参加、未出发。目标消息完整转述祁岚本人收到南汀青禾工坊 2026-10-25 周日浅浮雕课正式书面确认，回执 TQ-762-M，同样明确有效、未参加、未出发；两条都声明不是说话者本人的预约。人物、课程、回执、账户均为合成资料，使用不同的新来源 ID。

只在隔离评测进程设置支持的 MEMORY_LLM_QUEUE_SIZE=1、IDLE_SECONDS=0，生产参数不修改。在第一个真实作业进入 processing 后、调用原处理函数前用事件暂留；第二条请求执行期间实测 processing=1、inflight=1，发生真正的 capacity 跳过。没有设置非法容量 0、篡改计数或伪造模型输出。第二条返回后释放第一条，真实 writer 完成，最后 processing/inflight=0，作业终态明确。

## 根因与保留的失败

基线两条都得到实际 pro 的 200 回答“我记下了”。目标完成路径 memory_enrichment_scheduled=false，scheduler 真实记录 reason=capacity。目标只保留 Stage28 的 pending 身份摘要：body=NULL、terms=0、links=0，不在可见原话集合中；释放占位作业后仍然 pending，目标没有 claim。

完整目标输入已到达实际模型，原话缺失不是信息不全或模型规模问题。旧来源、旧 claims、旧聊天存档和父 fixture 均不变，保护检查 13/13；容量导致“答应记下却没有可检索原话”的持久化要求单独保留为失败。不能用聊天存档增加或身份摘要存在代替完整原话保存。

根因是完整原话的 capture 也在后台 _process_job 内；schedule 拒绝之前，根本没有机会执行。语义提取容量限制因此同时丢弃了合法原话。

## 唯一业务文件的服务器优先修复

在 CharacterContextService.complete_turn 的原有合规语义路径中，先用原认证 user_scope、来源 ID、完整正文与可信 received_at 调用仓储 capture_source，再提交有界语义任务。新增 source_capture 完成诊断，区分原话是否 recorded 与语义是否入队。capacity 仍然跳过语义提取，但不影响已完成原话的持久化和独立检索。

原数据库事务的摘要/身份/时钟核验、撤销及 owner fence 保留。返回 stale/revoked/conflict 则不再排语义任务；capture 抛错时记录 source_capture_failed，也不排队。缺失或不支持的旧适配器保留原流程。实际 writer 后续仍按同一可信时刻再次验证 source capture，不能绕过晚到撤销。

明确不要保存、敏感信息门禁、显式删除、被拒绝的备忘录/事实入口保持原规则。已有“假设原话可以保留为原话，但不能变成本人事实”的契约保留，不能为测试全绿把它改成强制拒绝或事实。关系和语义提取没有新增强制提取。没有修改 API、数据库 schema、推理模型、RAG 文档或原话权限。

对应失败检查还发现原保存异常日志有两个 %s 却只传一个参数，导致原始捕获异常时二次日志 TypeError。只修正这个完成路径的占位符，保存失败诊断现在可以正常返回；其他未涉及日志不做全文件整理。

## 原生复验与新冷检索

在同样的真实占位和容量 1 下，目标仍 capacity 跳过语义提取，但返回前已经 recorded 完整原话：摘要清除、词项 111、claim links=0；可见原话读取返回完整目标正文。释放占位作业前后状态相同，可信时刻与实际完成 receipt 一致，没有强制生成目标 claim，也没有越过容量提交目标 writer。

随后只新增一次完整指向朋友祁岚预约的冷问题，询问课程、日期、私人回执、有效性及是否参加/出发，明确区分说话者和朋友。仅该问题的历史读取观测返回 []，原仓储历史与存档不删除，完整原话已经真实保存，必要信息不缺失。

真实 prepare 的 raw_source_status=available，选中目标真实来源 ID，完整目标进入 episodic packet 和实际 pro 输入，未成为 system 指令。模型实际返回南汀青禾工坊浅浮雕课、2026-10-25、TQ-762-M、仍有效、未参加、未出发，并明确这是祁岚而非用户本人的预约。没有将私人回执伪作公开文档编号。容量修复、权限、完整输入、持久化和冷检索保存审计合计 35/35。

## 只执行相关检查

新增完成路径测试首轮 9 passed / 2 failed。一个是错误地要求所有假设转述都不能保留原话，与既有 source-only 契约冲突；改为明确核对允许原话而无事实，不改业务门禁。另一个是真实保存失败日志占位符问题，已修复。只复验这两项，2 passed / 9 deselected，累计 11 个相关检查通过；没有重跑通过的九项或旧身份/删除回归。

检查包含真实 SQLite 原话保存与 pending 时钟、容量跳过、无 target fact、opt-out/备忘录/假设转述、删除原文不入池、三种事务拒绝、capture 异常不入队、实际空语义 writer 的同钟复核，以及 capture 完成先于真实 schedule 容量检查。

新增测试和工具 Ruff 通过，character_context 的一项历史 import 格式诊断保留，对 HEAD 没有新增；格式校验 AST 等价，没有重复行为测试。成功门槛以保存证据运行六项控制：真实修复可接受，原始容量丢失、缺正文、强制本人事实、历史混入冷问题、错误回答均被拒绝。模型重放 0。

## 运行服务及证据

修复在服务器隔离入口有效后，备份唯一业务文件的新旧版本，只重启后端加载。新服务组 2049738、应用 2049742，就绪 200；真实进程仍 deepseek-v4-pro、65536 上下文、语义审核 30 秒、原话开关 true。私密 provider 配置、backend/.env、启动脚本、生产队列参数和其他服务进程不变，没有主动生产问答、显式业务数据写入或结构迁移。

原始基线在 `evaluations/r148pg.s3/stage29-source-capacity/`，复验和新冷问题在 `stage29-source-capacity-fixed/`。原始失败、检查、保存审计与备份在 `backups/backend-chain-20261001/stage29/`，运行加载/回滚准备在私密 `stage29-deployment/`。秘密与生产资料不进入 Git。

```text
python -m evaluation.native_mixed_context_probe --memory-capacity --capacity-cold-query --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage29-capacity-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

模式只接受已验证一次性集群，控制容量差异明确，仅一个新占位、一个完整目标和可选一条新冷问题，不是全套验收。它证明当前配置下合法原话不再依赖语义队列的可用容量，不保证所有语义任务成功、数据库永远可写、生成期间没有撤销或所有回答完全准确。Stage24 公开子任务遗漏仍保留，未证明其存在后端内容丢失。持续目标保持 active。
