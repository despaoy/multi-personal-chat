# 第二十五阶段：原话召回的真实跨用户隔离

日期：2026-10-01。基线 `db88f851f74783f33c912a9c09a0b294aa3935fe`。仅一个新原生生成请求，4 次 deepseek-v4-pro HTTP 200 / stop，正常 65536 上下文；旧来源写入、旧回答、公开文档导入和此前搜索回放均为 0。原生检查 44/44，保存请求独立审计 45/45；新增七项审计正反例 7/7。生产原话开关保持上一阶段用户明确授权后的 true，没有改生产业务代码、配置或重启服务。

## 完整来源与新账户

从原始检查全部通过的 `stage3_stage22_owner_only` 复制到一次性数据库 `stage3_stage25_cross_user_ready`，先检查 data_directory、源证明与索引逐文件 SHA。原源库保持原样，不使用后续保留失败的 Stage23/24 原始运行作为全通过前置。

已有账户 1 的本人确认 PB-681-Q、后来撤销、朋友季澄完整转述 FJ-394-L、确认当时有效/未参加/未出发，都是已验证的完整原生来源，不重新写入。公开资料仍保留三份完整工坊文档，未要求重答旧公开问题。这里所有账户、人物、回执、工坊和发言都是明确合成数据，无真实用户资料。

先只在复制库重设原合成账户随机密码，原生 login/me 均 200。随后清掉账户 1 Cookie，在隔离实例临时允许公开注册，正常 register/login/me 创建账户 2，三接口均 200，实际 role=user，未伪造 JWT 或把新账户升级为管理员。生产注册设置不修改。

新问题只要求从当前账户权限内既存记录核对朋友私人回执、确认、效力、参加和出发；如果当前账户没有，应说明无法核对，不能编造、用另一账户替代或用公开课程编号替代。问题没有 FJ-394-L 或 PB-681-Q。来源信息在账户 1 完整，账户 2 没有该来源是预设权限边界，不能把拒绝/缺失当成输入不全的提取失败。

## 真实入口伪造与实际模型输入

账户 2 的认证请求同时带账户 1 的 senderId、userId、conversationId、sessionId，sourceMessageId 故意复用账户 1 那条完整朋友原话的 ID。未注入模型 messages、客户端 history、源仓储或已有私人内容。

`/api/generate` 正常返回 200，实际准备结果 platform=web、adapter=web-character、sender_id=2、conversation_id=2、conversation_type=private，memory_scope_key 为当前账户 2 的私聊范围。私聊用户范围没有被伪造的 sessionId=1 改成账户 1。没有做普通历史消融；账户 2 实际仓储历史、claim 候选、选中 claim 均为空。原话读取确实执行，status=no_match、无匹配来源、episodic context 为空，而非关闭读取导致伪通过。

检查全部 4 次真实模型请求的全部消息，而非仅检查最终回答：没有账户 1 的完整三条私人陈述、FJ-394-L、PB-681-Q。请求预算分别为 768、160、1024、768，涵盖语义审核、策略、回答及完成后 writer；空 claim 选择路径不必产生虚构的 selector 调用。没有错误本地小模型请求。本次完整问题到达实际 pro 回答输入，真实模型回复与 API 文本完全一致：无法从当前账户记忆核对这些私人内容，当前账户没有对应原话，不能确认。没有返回另一个账户的私人编号。

## 写入归属与存储核验

模型完成后真实后台写入路径只捕获账户 2 当前问题。复用相同 sourceMessageId 的两条消息分别保存在各自账户范围，各恰好一条，来源 scope key 不同。账户 1 朋友来源仍 recorded 且全文不变；账户 1 其他来源、两条本人 claims、物理聊天存档 SHA 前后完全一致。源 fixture 数据库的 memory_sources、memory_source_terms、character_memories、messages 全表确定性摘要也前后一致。没有把账户 2 提问覆盖到账户 1 那条原话。

只复用 Stage23 已保存的本人权限内成功读取证据，不重播已通过的正例；本阶段证明的是这一真实跨用户、同来源 ID 组合边界，不是所有入口、分支、角色或会话类型的通用权限证明。

## 新增审计与工具准备记录

现有原生探针新增互斥 `--memory-scope`，自动启用原话读取以匹配当前生产配置，仅执行最后一个新问题。复用 Stage22 前三个完整来源并验证来源全文确实保存；正常认证新用户、观察实际 UserScope、全部模型输入及前后 SQL 摘要。普通历史不消融，与现有冷读模式明确区分。保存请求独立审计移除没有搜索请求时的 vacuous“搜索成功”项，改成不重播搜索，增加原始模型文本与 API 一致项，45/45；原始 44/44 文件仍保留。

七个新增审计正反例覆盖：合法公开内容、system/user/assistant 任意角色的私人泄漏、非回答的 writer/selector 调用泄漏、HTML 编码私人回执、没有编号的完整原话泄漏。只执行这个新测试文件及三份相关 Python 工具的 Ruff，全部通过；没有全套回归或额外云审计调用。

测试准备曾两次在生成之前停止：第一次临时脚本在备份目录相对解析 fixture 路径失败，数据库未创建；第二次只读源快照误用不存在的物理表名 character_memory_claims。核实当前 schema 的真实表为 character_memories 后修正。两个已终止任务、原脚本、日志和证明独立保留，实际模型/生成调用均为 0；第二次只创建隔离副本和完成正常测试账户认证。修正后的新目录执行一次真正生成，没有对已通过问答反复重试。准备工具错误不能声称为软件业务链路缺陷。

## 复现与生产状态

```text
python -m evaluation.native_mixed_context_probe --memory-scope --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage25-scope-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

该模式只接受已验证的一次性集群路径，不是生产数据库迁移。运行证据在 `evaluations/r148pg.s3/stage25-cross-user-ready/`；独立审计、前两次准备失败记录与七项新测试在 `backups/backend-chain-20261001/stage25/`。不保存认证秘密或生产原话到 Git。

生产组 2017475、应用 2017480，原话开关 true、就绪 200；六份相关业务代码、backend/.env、启动脚本与私密 provider 配置均保持上一阶段状态。生产生成请求和显式业务数据写入均为 0。本阶段没有发现这个边界的后台缺陷，因此没有人为修改已正确的核心实现。

Stage24 的公开子任务遗漏仍保持真实失败：完整公开资料在实际 pro 输入，原始回复与 API 一致，目前没有后台丢失证据。没有重复那个问题挑选成功回复，也没有将该遗漏宣称已修复。下一阶段检查同用户不同角色的新增范围边界，继续只测对应的新链路。
