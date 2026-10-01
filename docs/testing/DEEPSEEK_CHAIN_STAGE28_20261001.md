# 第二十八阶段：新来源标识的并发消息只能绑定一条正文

日期：2026-10-01。基线 `e3479dfe1a0171da0d60350692ae4eeb0c32ac59`。本阶段仅新增一个同账户、两个客户端会话、同一新来源标识的并发边界，然后只复验这次失败。共四个原生请求，三个 200 回答、一个 409；实际 deepseek-v4-pro 调用 14 次，基线 9 次、修复 5 次，均 HTTP 200。被拒绝的请求没有模型调用。保存结果重审没有新增模型调用；没有重复旧写入、删除、问答、搜索、文档导入或全套回归。

## 完整输入和隔离方式

复用 Stage22 原生已验证全部检查通过的 `stage3_stage22_owner_only`，完整首三个本人确认/撤销、朋友转述逐项相同。分别克隆基线与修复复制库，核对一次性集群 data_directory、三个文档索引的逐文件 SHA 和父数据库四张表摘要。只在复制库重设合成账户密码，通过正常 login/me；普通历史保持真实仓储投影，没有消融。

新输入 A 完整转述朋友田眠：东沙杉墨会馆 2026-10-17 周六苔景绘画课正式确认，私人回执 RA-428-D，预约有效、未参加、未出发。B 完整转述另一朋友林朔：南溪青栎工坊 2026-10-18 周日陶片拼画课正式确认，回执 RB-905-N，同样完整说明有效、未参加、未出发。两条都明确被描述者不是用户本人。人物、课程、回执和账户均为合成数据。

两条不同完整正文故意共用 `web:stage28-fresh-concurrent-identity`，使用不同客户端 sessionId；认证账户、角色和私人来源 scope 相同。该标识在复制库初始确实不存在。观测栅栏等待两次实际验证结果后一起释放，放大验证与模型之间的竞争窗口，不跳过或替换真实验证。不同 sessionId 本来就可由两个浏览器会话同时提供。

## 实际根因及保留的失败

排队串行键包含客户端 sessionId，而私人记忆实际归账户。同一来源身份的两个会话都可以读到 new，Stage27 的只读快照不能为新标识建立不可变正文约束。

基线两条都通过真实入口并返回 200，实际 pro 分别回复“我记下了”。最终 scheduler 第一条 source_capture=recorded、accepted=1、persisted=1；第二条 source_capture=conflict、reason=source_conflict、accepted=0、persisted=0。源记录只保留 A 完整原话，底层防覆盖正确，但第二条已经成功回答、仅增加聊天存档，未保存为该来源。

旧来源、旧 claims 和旧聊天存档全部保持一致。新 writer 合法新增一条完整证据的 quoted_source 记录，described_subject=not_resolved，没有将朋友硬转成本人事实。原始审计误用整个 claims 集合相等，得到 13/14；按原有行 ID 和所有字段核对旧记录不变后，来源保护为 14/14。原始文件不覆盖，仅重审保存结果，模型重放 0。入口“不同正文不能同时以同一新身份获得成功回答”的失败单独保留，不能用来源保护全绿掩盖。

## 服务器优先修复

新增共享 reserve_plan，在模型调用前的真实数据库事务中取得既有 owner fence 锁，重新查精确 owner/scope/source identity，再绑定新来源。SQLite 使用 BEGIN IMMEDIATE，PostgreSQL 使用同一个 owner 行锁；无需依赖单进程内存锁或客户端排队键。

memory_sources 新增可空 body_digest 字段。pending 绑定只有当前清洗后完整正文的 SHA-256 和服务器接收时间，body=NULL，不可供原话读取或 RAG 使用；它不生成事实、不是内容哈希作为来源标识，也不是全局正文去重。原始身份始终由明确 scope 与客户端/服务器消息 ID 构成。

同一身份的另一正文原子地返回 conflict，入口 409 / source_identity_conflict。相同正文的未完成重试复用第一次可信接收时刻，生成失败后也不允许另一正文冒用这个已绑定身份。绑定不是租约，没有超时后替换正文的窗口；未完成绑定仍是 pending，直到同正文成功写入或明确删除。

服务器时刻通过 MessageRequest 的私有属性交给 TurnInput，再用于真实上下文准备、候选验证和后台 source capture。客户端 JSON 无法指定该私有属性，响应/模型序列化不暴露它。最终 capture 核对 pending 摘要及同一接收时刻；正文或时刻不一致则拒绝，完成后清除冗余摘要。已 recorded 的正文和时刻不可覆盖，最终事务撤销及时间 fence 继续生效。

明确不要保存、敏感信息门禁拒绝的内容、删除指令不创建 pending 绑定，沿用原 memory_write_allowed 和删除识别。原话删除和 scope 清空同时清除 body_digest；owner fence 后的旧 pending 工作也不能用新的 worker 时钟复活。既有 digest=NULL 的 claim-link pending 锚点兼容首次填入。

这是来源身份绑定，不保证模型处理期间没有新的撤销操作，也不保证所有语义 writer 作业都能成功排队或提交。模型失败留下的 pending 不可检索，没有原文或事实；后续需继续核对保存容量、失败反馈和交付链路。不能以本阶段通过宣称长期记忆、RAG 与动态上下文全部无误。

## 原生复验和对应检查

相同两个完整输入、相同并发栅栏和正常认证，修复后一条 200、一条模型前 409。只有 A 准备并调用模型，B 全文没有进入任何真实模型请求；只有一次 scheduler 记录、一次聊天存档、一次 source capture，原话与 A 完全一致且最终摘要清除，没有晚到 conflict 作业。旧来源/旧 claims/原存档及父 fixture 摘要不变。保存结果审计 26/26，实际来源保护 14/14。

只执行改动的入口文件和新增绑定测试，首轮 24 passed / 2 failed。两项失败是测试直接插入锚点、更新 fence 后没有提交准备事务，导致嵌套 BEGIN；修正准备后只执行这两项，再新增旧库可空字段升级、scope 清空两项，4 passed / 11 deselected。累计 28 个对应行为检查通过，已经通过的 24 项未重复。覆盖真实双线程竞争、同正文失败重试时刻、完整 capture、错误正文/时钟、删除摘要、迟到 capture、另一账户、旧 pending、opt-out/删除不绑定、读取/预约失败在模型前拒绝、私有时钟与真实准备传递。

新增较小文件 Ruff 通过。旧 SQLite 52、PG 177、models 103、character_context 1 项历史诊断均保留，对 HEAD 无新增。新增字段类型写法的一项诊断修正后通过。并发工具从实际测试主体提取为独立 helper，只做参数名称对应，AST 核对一致；格式与导入整理后 AST 等价，不重复原生请求。另以保存结果检查成功门槛：真实复验通过，原始双成功、错误冲突码、拒绝正文混入模型、晚到 conflict 回执、错误存档均被拒绝，6/6，无新模型或原生调用；--require-success 现在核验拒绝与保存反馈，而非仅来源防覆盖。

## 生产加载和证据

隔离验证通过后备份八处旧/新业务代码，只重启后端。新服务组 2042662、应用 2042666，就绪 200。实际仍 deepseek-v4-pro、65536 上下文、语义审核 30 秒、原话开关 true，私密 provider 配置、backend/.env、启动脚本和其他服务进程不变。

生产使用 PG_HOST/PG_USER/PG_PASSWORD/PG_DATABASE 等分项配置。首个重启前保护误假设必须有 DATABASE_URL，在停止服务前退出；原服务保持运行。按项目现有 URL resolver 取得实际连接后重新校验。初始化只增一个可空字段，原 sources/claims 的 SQL 内聚合摘要及数量前后完全相同，body_digest 非空数量为 0。没有主动生产问答或显式业务数据写入，没有读取/输出生产原话正文或凭据；数据库内的摘要用于完整性证明。

基线与复验分别保存在 `evaluations/r148pg.s3/stage28-source-race/`、`stage28-source-race-fixed/`；原始检查、独立审计、代码备份在 `backups/backend-chain-20261001/stage28/`；结构前后摘要、运行加载与回滚准备在私密 `stage28-deployment/`。秘密和生产资料不进入 Git。

```text
python -m evaluation.native_mixed_context_probe --memory-race --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage28-race-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

该模式在明确验证的一次性复制库仅发两个完整新请求，不是生产迁移或全套验收。Stage24 公开子任务遗漏仍保留，未证明后端内容丢失；持续目标保持 active，下一阶段只增加尚未验证的真实链路边界。
