# 第二十三阶段：无结构化 claim 的朋友原话能否读回

日期：2026-10-01。基线 `eaec12acf12e42617df9b761e08db5f5d40bdad3`。一个全新朋友问题，先用生产当前默认设置验证，再仅复验这个失败问题的召回开关差异。合计两个原生生成、10 次 deepseek-v4-pro HTTP 200 / stop；旧写入、旧回答、文档导入和全套回归回放均为 0。

## 完整输入与真实默认配置

上阶段已通过的隔离原生 fixture 中，朋友季澄的完整转述包含：已收到书面确认、朋友私人回执 FJ-394-L、当时预约仍有效、未参加课程、未出发、描述主体属于朋友而非说话者。后台 writer 合法 NOOP，但原话及来源 ID、观测时间实际保存并建立源词面索引。

这次不重写原话，只提出新的朋友问题，要求从此前完整转述恢复这些字段、明确不能当作独立核验后的最新状态，且不列本人回执；另外查公开课程编号及普通雨天规则。问题不含正确私人编号、确认/效力/参加/出发答案。

分别从已验证的 `stage3_stage22_owner_only` 创建新数据库 `stage3_stage23_friend_default`、`stage3_stage23_friend_enabled`。先核验一次性集群 data_directory，复制索引逐文件 SHA 一致。在副本重设原合成账户随机密码，正常原生登录及身份接口均 200。副本保留最初账户名 stage20-cold-fixed，它与复制来源运行名不同，不能因复制数据库而改身份或伪造 JWT。源数据库及旧证据不修改。

仅新问题做普通历史消融：记录真实仓储原历史，再返回空历史；完整朋友旧发言仍存在原历史和原话仓储，没有删除消息、重构模型 messages 或重播写入。两次副本问答前的来源快照、claims、索引 SHA 逐字段相同。

再次核对生产进程初始环境及 dotenv：原话召回未设置，构造器默认 false。第一组显式 false 匹配这个当前设置。生产大模型本身仍是 deepseek-v4-pro，上下文 65536；没有用小模型制造对照失败。

## 根因与因果复验

默认组原生检查 29/38：完整朋友原话已持久化，但准备诊断 source status=not_checked，没有原话召回记录或 dialogue_evidence，朋友字段未进入实际模型输入。结构化选择未选任何本人 claim 给朋友问题是合理的，不能强制把朋友转述变成本人事实。

真实默认回答说没有季澄原话依据，不能判断确认、编号、参加或出发；公开编号 YY-573-R 和雨天规则仍正确。缺失发生在后端未启动原话读取，不能把模型面对缺失输入的保守回答认作模型能力不足。这是默认配置尚未提供该 source-only 冷读取能力的实际覆盖缺口，而非第三方提取 NOOP 本身出错。

第二组只把原话召回设 true，并如实修改对应诊断值。保存脚本归一化后与默认脚本完全一致；没有改 fixture、提问、画像、模型、来源数据库、claim、索引、历史处理或业务代码。原话 scope 检索、当前授权重读、完整包编译实际执行，source status=available；4 条被送达，另 1 条候选被省略。完整朋友原话和原 ID、观测时间进入实际回答请求，包仍标记历史用户发言、说话者为 user、描述主体和当前有效性未解析。

真实回答恢复 FJ-394-L、已收到书面确认、转述时仍有效、尚未参加、没有出发；明确“只是你对季澄本人预约的转述，不能当作独立核验后的最新状态”。没有报本人 PB-681-Q，没有将原话放进 system 或公共知识证据。公开编号与普通雨天规则只有课程 doc_1_chunk_0 引用。两组问答前后都没有增删或覆盖 claims。

## 审计修正与相关检查

开启组原始检查 37/38，唯一失败是确认状态审计将“收到”到“确认”限制为 10 字；真实答案之间有完整课程名称。因此只修正审计：在单个有界分句内允许完整课程名，拒绝“未收到”“无法判断”“是否收到”等不确定/否定表达，不把另一分句的“没有出发”误认为否定收到确认。该显式语句规则并非任意自然语言语义证明。

只跑 10 项新状态审计正反例，10/10；检查同一保存请求得开启组 38/38，默认组仍 29/38，不修改或洗掉真实默认失败。另 11 项因果控制通过，涵盖唯一开关变化、完整问题与来源快照一致、正常认证、相同原生 claims/索引、没有写入变化及实际大模型请求。新增审计云请求 0。原始两组请求、结果和日志保留不覆盖，未重测之前通过的用例。

## 可复现模式与生产未解决事项

把朋友模式加入现有原生探针，没有另复制整套脚本。`--memory-friend` 只执行最后这个新问题，复用 stage22 已验证来源运行，要求前三条前置陈述相同及完整朋友源已持久化；复用前仍检查源运行全部通过。`--enable-source-recall` 只允许在朋友模式使用，默认组 false，实验组 true；既有历史模式的读取设置保留，本人模式仍保持上阶段的独立开启验证。

```text
python -m evaluation.native_mixed_context_probe --memory-friend --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage23-default-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt
python -m evaluation.native_mixed_context_probe --memory-friend --enable-source-recall --root /home/boot/lhm/multipersonal-runtime/evaluations/r148pg.s3 --run-label stage23-enabled-new --api-key-file /home/boot/lhm/multipersonal-runtime/config/deepseek-evaluation-api-key.txt --require-success
```

默认组故意保留尚未提供的原话读取能力，不能要求该组 require-success。该工具需要一次性集群里的已验证源数据，不是生产数据库迁移；认证秘密不进入 fixture 或请求记录。

生产默认缺口目前未修复，未开启生产配置。既有 [原话上下文契约](../architecture/MEMORY_SOURCE_CONTEXT_CONTRACT_20260927.md) 和 [读取对照记录](MEMORY_SOURCE_READ_COMPARISON_20260927.md) 列出的上线工作包括 source-only 真正可执行的遗忘、旧源索引覆盖以及完整联合验收。仓库现在已实现相关删除接口和定向测试，不能仅依据旧文档就声称实现缺失；也不能仅有 SQLite/stub 用例就声称当前服务器真实 DeepSeek/原生入口已经验证。下一步以这个完整 NOOP 朋友 fixture 验证原生独立删除及冷读不复活，再核对实际旧索引覆盖，逐项补证或修复后决定配置发布。

原始证据分别在 `evaluations/r148pg.s3/stage23-friend-default/`、`stage23-friend-enabled/`，独立审计和新测试在 `backups/backend-chain-20261001/stage23/`。生产服务组 1987299、应用 1987303 未重启，就绪 200；六个相关业务文件及三个配置与基线一致。生产生成请求和生产数据库操作为 0。本阶段定位了实际覆盖缺口，完成隔离因果验证，未宣称长期记忆整体无误或生产已解决。
