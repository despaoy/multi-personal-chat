# 第二十六阶段：已知来源冲突在回答前明确拒绝

日期：2026-10-01。基线 `424b79e09fbbd40ab6b0b07e27cc098abd1255ef`。本阶段只有一个新来源冲突用例，以及修复后对这个失败请求的一次定向复验。共两个原生请求：基线一个 200 生成回答，修复复验一个 409 拒绝；实际回答生成 1 次、deepseek-v4-pro 调用 3 次，均 HTTP 200 / stop。复验与保存请求审计额外模型调用均为 0。未重播旧写入、旧回答、文档导入或搜索，没有全套回归。

## 完整输入及既有契约

在已验证一次性集群核验 data_directory，分别从原始检查全通过的 `stage3_stage22_owner_only` 克隆 `stage3_stage26_source_collision` 和 `stage3_stage26_source_collision_fixed`，索引逐文件 SHA 一致，源 fixture 数据库不修改。仅复制库重设原合成账户密码并正常 login/me，均 200。

同一账户 1 已有完整本人 PB-681-Q 确认与撤销、朋友季澄 FJ-394-L 确认仍有效/未参加/未出发的原生来源。新陈述完整更正朋友的当前状态：工坊正式撤销其预约、FJ-394-L 作废、目前没有有效书面确认，仍未参加/未出发，明确朋友与本人区分、公开课程仍举办、不是删除指令。

该新陈述故意复用朋友旧来源 ID `web:0d126b81fcf4460eadbc8a3d9476f708`。内容不同、接收时刻不同，因此不是同文本同来源的合法幂等重试。所有人物、账户、工坊、回执及陈述都是完整合成数据。普通历史来自真实仓储，没有客户端注入或消融。

既有 capture_plan 契约禁止覆盖已 recorded 原话和观测时间，也禁止复活 revoked 来源；冲突必须跳过语义 writer 与事实写入。不能为通过用例而把新内容覆盖到旧 ID，或将朋友转述强制提取成本人事实。

## 根因：防覆盖成功，但响应先于冲突检查

基线原生入口返回 200，真实 pro 回答“我按你说明的顺序记下”“先前确认与后来撤销的顺序我会保留”。完整当前陈述、权限内旧历史与原话已经送达模型，输入信息没有遗漏，也不是小模型问题。

回答后完成路径的真实 scheduler 捕获来源，得到 source_capture=conflict、status=skipped、reason=source_conflict、accepted=0、persisted=0，没有语义 writer 调用或新 claim 关联。旧来源正文、时钟、索引、claims 全部保持原样，新陈述仅增加一条聊天存档，没有成为该旧来源的长期记忆。

所以底层防覆盖契约确实有效，独立源保护审计 35/35；但它发生在成功回答之后。入口让一个已知与原来源身份不相容的请求先生成“记下、保留”回复，随后后台拒绝写入。基线“已知冲突应在回答前明确拒绝”的入口一致性要求保留为失败，不能以底层检查通过宣布这个请求已成功存入长期记忆。

## 服务器优先修复与原生复验

在 ChatGenerationService 增加可选请求验证钩子，置于相同文本清洗之后、具体模型处理器之前。HTTP 服务构建时注入当前应用数据库的已知来源检查；不会意外使用另一个应用的全局数据库。

对 web-character 的角色请求，在认证账户对应的原话 scope 内，只读取本次精确来源 ID 的一个当前可见 recorded 来源。已有原文与清洗后的当前正文不同，返回 HTTP 409、稳定 code=source_identity_conflict，并明确提示新消息应使用新标识。不返回旧原文或私人资料。查源失败返回 503，无法验证时不继续生成。已存在的同文本重试、新来源及没有该权限范围记录的请求继续允许；stateless 管理入口和分支既有权限流程保留。

实际复验只发送同一个失败陈述和旧 ID，正常原生登录，HTTP 409 在模型准备前返回；无准备结果、无生成请求、无 scheduler 作业、无新聊天存档、无事实/关联/来源修改。旧朋友全文、时钟、旧聊天摘要及源 fixture 四张表摘要保持一致。独立保存请求审计 20/20。

该检查针对当前可见的既有来源；事务 capture 仍是最终保护，包括并发竞争、撤销、旧时间工作。当前验证没有证明所有新标识并发竞争或已撤销锚点的入口反馈都已覆盖，不能撤掉底层保护或宣称全部来源身份行为已完成。

## 只执行对应改动检查

八项新入口测试 8/8：已知冲突在处理器前拒绝、同文本重试、新标识、认证范围与应用数据库注入、读取失败 503、stateless 管理兼容、分支原权限、相同清洗文本比较。新测试使用独立 test_web_source_collision.py；已有 test_web_source_identity.py 由存在保护保留，未覆盖也未重跑。

只执行 test_memory_scope_audit.py 的八项新增 collision 控制，8 passed / 7 deselected，之前通过的跨用户七项未重测。覆盖真实匹配回执、不同 ID、非零 accepted、非零 persisted、其他跳过原因、缺失回执，以及内存空 tuple / JSON 空 list 与缺失集合的区别。六个相关改动 Python 文件 Ruff 通过。审计格式调整经 AST 等价校验，没有重复运行已通过测试。

原始基线 34/35 的唯一阈值失败，是把已有十条正常仓储历史误作“长历史回放”；改用 fixture 的零历史模板证明没有新增长历史注入，不删除已有对话去迎合长度。原始修复复验 19/20 的唯一审计误判，是内存 recent_results=() 与 [] 不相等；实际 JSON 与进程状态都没有 writer 结果。修正类型明确的空集合审计后重审保存结果得 20/20。两份原始结果及全部实际请求保留，不重发模型或 HTTP 问答，不掩盖基线入口问题。

## 生产加载及证据

有效修复先在服务器隔离入口验证，随后备份旧/新两份业务文件并重启后端加载。新服务组 2028818、应用 2028822，就绪 200；实际运行环境仍 deepseek-v4-pro、65536 上下文、语义审核 30 秒、原话开关 true。provider 私密配置、backend/.env、启动脚本和其他服务进程保持不变。未主动发起生产问答或业务数据写入。

原始基线在 `evaluations/r148pg.s3/stage26-source-collision/`，修复复验在 `stage26-source-collision-fixed/`；独立审计、测试和原始代码备份在 `backups/backend-chain-20261001/stage26/`，重启与回滚准备证明在私密 `stage26-deployment/`。认证秘密、生产原话和运行时配置不进入 Git。

```text
python -m evaluation.native_mixed_context_probe --memory-collision --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage26-collision-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

模式只接受已验证一次性集群，复用 Stage22 完整资料，仅执行新来源冲突请求；409 分支核验不生成与不写入。它不是生产数据库迁移或全套验收。

Stage24 的混合公开子任务遗漏仍保留，未声称修复。下一步定向检查已经撤销的同账户来源标识重放，核对不可见锚点状态与入口反馈是否也一致，不重新测试已通过的 source-only 删除或全部聊天用例。
