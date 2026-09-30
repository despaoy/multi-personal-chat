# r123：主体元数据贯通与一次被否定的模型输入变更

## 确认的链路问题

r122 在数据库 metadata 保存了 quoted_source/speaker_role/described_subject，但
MemoryItem 未携带该属性。检索/直接仓储读取丢掉区分；完整证据包固定
subject_scope=current_user_not_character，普通参考声明也要求按用户本人理解。
这与原话可能描述妹妹/同事的表示契约不一致。

另重放 r122 的姐姐公司问题：完整原话已在实际模型输入中，只有一次生成，
原始模型输出=guard结果=HTTP回复，均冒领为“我”。没有发现后处理或接口拼接改错答案。
既有r86/r87/r90/r91已做位置、人物上下文、思考与量化对照，本轮未重复整套。

## 保留与撤回

保留应用生成的 source_observation 标记，经真实仓储、MemoryItem、混合检索和
compiled.memory_packets 传递；直接仓储读取观察记录也保留 evidence/source IDs。
普通模型的 attributed_to 标签不能单独生成这个标记，个人事实默认不受影响。

候选显示把观察条目标为“发言者=用户、描述主体=未解析”，混合参考区不再一概称本人事实，
完整JSON使用not_resolved并区分speaker。真实复测发现退步，因此**不启用为默认显示**。
compile_reference_context(observation_semantics=True) 仅供显式实验/测试；默认false保持
r122模型输入表示，标记本身不丢。没有新UI开关、模型复核或人物system改写。
因此不能说默认参考区的主体语义矛盾已彻底修好；下一轮必须改为独立的原话通道表示。

## 真实证据

候选两场景（妹妹/本人，赠礼偏好）共4HTTP200，4新generation、4录制独立writer输出、
0新writer。真实Qwen3-8B-AWQ无LoRA、冷读、隔离SQLite、source recall显式开启、RAG关闭，
测试身份与inline排队。妹妹问题仅复述问题，赠礼正确。

固定候选妹妹请求，保留system、问题、所有source packet和采样；三个视图×3seed：

| 视图 | 新生成 | 回答 | 实际输入token |
| --- | ---: | --- | ---: |
| 新主体未解析显示 | 3 | 均复述问题，没有给出归属 | 2019 |
| 同一记录的旧显示 | 3 | 均正确区分妹妹与用户 | 1990 |
| 删除重复观察参考、完整原话包仍在 | 3 | 均正确区分妹妹与用户 | 1600 |

9次全stop，远小于8192，未增加writer/检索/数据库写入。删除实验仅在source id、原文、
接收时间能独立覆盖观察记录时允许；不是无条件清空记忆，不是自动将用户原话转成连续history。
source_only还保留原来的空参考容器和system，因此不是最终产品实现。
这只是一个固定场景，不能把3/3推广成完整source-routing方案已验证。未默认删除参考区。

## 测试与产物

- .tmp/mechanism-r123-observation-{traces.jsonl,manifest.json}：4次真实HTTP候选。
- .tmp/mechanism-r123-ablation-{results.jsonl,manifest.json}：9次固定输入生成。
- .tmp/mechanism-r123-regression.xml：候选828项通过、4既有警告。
- .tmp/mechanism-r123-final-focused-valid.xml：撤回默认显示后21项通过。
- .tmp/mechanism-r123-final.xml：最终831项通过、3198未选、4既有警告；上述聚焦组重叠不累加。
- 真实SQLite证明标记经过repository→memory service→实验完整编译；普通事实、混合包、
  已过滤记录不变，用户资料仍只在不可信user区。新增默认不启用回归显示的断言。
- 初始测试误读load_relevant_memories的(tuple,count)契约、一次错误测试文件路径、撤回时
  缺replace导入均已修正；没有绕过相关断言。repository已有31处UP045风格问题未做无关重写。

远端r123快照保存当时候选默认显示，当前本地默认已撤回；生产未部署/重启/迁移。
所有远端运行结束SSH关闭。更强模型隔离评估已向用户询问，未下载或替换模型；等待回复
不妨碍继续做原话通道与重复证据改进。目标保持active。
