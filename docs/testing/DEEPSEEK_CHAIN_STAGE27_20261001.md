# 第二十七阶段：撤销来源重放在回答前拒绝

日期：2026-10-01。基线 `b3a617991469a44b81124b4399e5853d8b584fac`。仅执行一个新的同账户撤销来源标识重放请求，以及服务器修复后对该失败请求的一次定向复验。两个原生请求：基线 200 生成一次回答，修复后 409；实际 deepseek-v4-pro 调用 4 次，均 HTTP 200。修复复验及保存结果重审均没有模型调用；没有重跑旧写入、删除、问答、文档导入、搜索或全套回归。

## 输入完整且复用已经验证的撤销状态

从一次性集群的 `stage3_stage24_source_erasure` 分别克隆两个隔离数据库，核验 data_directory、索引文件 SHA、源数据库四张表摘要和精确权限范围。原账户只在复制库重设密码，通过正常 login/me；没有注入生产资料。

复用门槛明确检查 Stage24 独立 source_erasure_gate 43/43，并直接核对撤销锚点：state=revoked、body/observed_at=NULL、terms/links=0、原始物理存档保留一条、本人其他来源和 claims 保留。Stage24 原始 40/46 和联合 43/48 的公开子任务遗漏仍是未通过，不能用来源删除门槛代替整份结果全绿。

新请求完整重发最初朋友季澄的预约陈述：北岸榆影工坊周日纸雕课已经书面确认，私人回执 FJ-394-L，预约仍有效、未参加、未出发；明确是朋友的预约而非说话者本人。人物、工坊、回执、账户均为合成数据。完整原话与首三个既有陈述逐项校验，仅复用已撤销的原 sourceMessageId，模拟旧客户端延迟重试；没有缺失必要信息。当前请求重新提供的完整文字可以送达模型，不能误报为从已撤销历史泄漏。

该用例只禁止已撤销消息身份复活。未来用户真正重新分享、使用新来源标识的完整陈述仍应允许；不能把来源撤销变成永久禁止用户提供同一段文字。

## 根因和原始结果

基线正常认证同一账户，当前可见来源读取不包含已撤销朋友原话，普通历史按既有授权过滤，锚点没有复活。实际 pro 却返回 200，回复“我按你转述的内容记下”。随后真实 scheduler 的 source_capture=revoked、reason=source_revoked、status=skipped、accepted=0、persisted=0，没有语义 writer 调用。

基线来源保护审计 30/30，说明底层拒绝复活有效；入口一致性要求“已知撤销身份在模型回答前拒绝”单独保留为失败。不能用来源保护全绿宣布成功记住了这个请求。只有聊天存档增加一次重试，旧原始存档 SHA、来源、claims、父 fixture 数据库保持不变。存档计数用请求前最大行 ID 区分旧原话和当前重试，避免同标识同正文被重复计为旧资料。

Stage26 入口只查当前可见 recorded 原话，撤销锚点被过滤后查不到，因此误当作新标识放行。问题在后端入口与最终保存反馈顺序，不是模型规模或输入信息不足。

## 服务器优先修复

新增共享 admission_plan，在一次 SQL 快照内查精确 authenticated owner、scope、source key 的状态、观测时刻、owner fence，并在 SQL 内比较当前完整正文是否相同。只返回 new/pending/recorded/conflict/revoked/stale 状态，不读出旧正文，不做全局来源 ID 查询，也不写数据库。SQLite、异步及同步 PostgreSQL 包装使用同一计划。

现有清洗之后、模型处理器之前的应用数据库验证改用该状态计划。revoked/stale/conflict 返回 HTTP 409 和稳定 source_identity_* code；读取失败或未知状态返回 503；清洗后空消息 422。新标识、pending 锚点、同文本 recorded 重试继续允许；stateless 和分支的原流程保留。用户身份来自实际认证账户，不能信任请求伪造 sender/user。

最终事务 capture 的 owner lock、不可变原话、撤销及时间 fence 保护保持原样。入口是模型调用前的快照检查；模型处理期间发生的新并发撤销或新标识竞争仍由最终事务保护，本阶段不宣称解决所有并发反馈问题。

修复后只重发原失败请求：正常认证，原生 409 / source_identity_revoked，未准备模型、未生成、未排 writer、未新增聊天存档，实际模型调用 0。撤销锚点前后完全一致，原存档一条、旧源/claims/父 fixture 摘要不变，独立保存结果审计与原始检查均 17/17。

## 仅检查修改涉及部分

两个相关测试文件首轮 24 passed：更新的入口已知冲突、正常重试、新 ID、认证 scope、读取失败、应用数据库、stateless/分支、清洗比较，以及新增 revoked/stale/pending/未知状态；真实 SQLite 覆盖来源状态、正文冲突、时间 fence、四种其他权限范围、单条只读 SQL 和空正文。

随后新增清洗后空消息保护，只执行该新增测试：1 passed / 12 deselected，没有再次运行之前通过的十二项入口检查。累计对应修改检查 25 项通过，没有全套或旧 source-only 删除回归。格式调整校验 AST 等价，不重复测试。

新增/较小修改文件 Ruff 通过。旧 database.py 的 52 项、pg_database.py 的 177 项历史格式诊断与 HEAD 对比无新增，未为本次来源检查重写整个旧数据库文件。工具导入格式的一项诊断已修正，AST 不变；保存审计重算不调用模型。

## 运行服务和同步证据

有效代码在服务器验证后备份旧、新四份业务文件，再重启仅后端加载。新服务组 2035881、应用 2035885，就绪 200；真实进程仍 openai_compat / deepseek-v4-pro / 65536 上下文 / 30 秒语义审核 / 原话开关 true。provider 私密配置、backend/.env、启动脚本及其他服务进程不变，没有主动生产问答或显式业务数据写入。

实际生成记录在 `evaluations/r148pg.s3/stage27-source-replay/` 与 `stage27-source-replay-fixed/`。独立审计、25 项检查、差异 lint 和原始代码在 `backups/backend-chain-20261001/stage27/`；运行加载与回滚证据在私密 `stage27-deployment/`。认证秘密、生产原话、运行配置不进入 Git。

```text
python -m evaluation.native_mixed_context_probe --memory-replay --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage27-replay-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

该模式只复用明确验证的隔离撤销 fixture，执行一个新重放请求，不是生产迁移或全套验收。Stage24 完整公开输入送达但回答遗漏的问题仍保留；目前未证明那里存在后端内容丢失。下一阶段只新增尚未验证的链路边界，不重跑本阶段已通过用例。
