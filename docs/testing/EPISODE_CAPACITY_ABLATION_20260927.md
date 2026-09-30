# 原始记忆读取的模型能力与上下文干扰诊断

2026-09-27，r90。96次真实无LoRA推理；不是生产HTTP端到端测试，没有运行写回、输出guard或重试。默认实现不变。

## 方法与边界

复用r86的8个预先定义案例，保持问题、原话、采样参数（temperature=0.7、top_p=0.9、max_tokens=2048、thinking=false）和两个seed不变，不根据回答挑选证据。全部第一轮模型请求已冻结，未使用原始guard重试输入。

- framed_persona：原完整system，原始user/assistant按聊天帧重放，保留来源ID/时间元数据与当前问题包装。
- framed_plain：只去掉上述system，其余完全相同。
- natural_persona：原完整system，原始user/assistant聊天帧，不加元数据，末尾为原问题。
- natural_plain：上述自然聊天帧，无system。
- core_persona/core_plain：在natural对应视图上，仅移除seed_cases生成的30轮固定段落编号干扰。用原fixture的全部episodes逐字段匹配前缀，尾部30条的双方原文、时间、session全部核对；任何改动即拒绝删除。不读rubric挑证据，不是产品相关性过滤器。

前4项共64次，后2项共32次。persona标签实际代表**完整system**，包含人物设定、动态关系、能力与检索策略，不能把差异归因于某一段人物提示词。natural同时移除来源元数据和问题包装，只能评价组合；不用于时间推理或生产部署。去掉system的控制组不是合格的人物产品。

## 逐案例人工检查

以下均检查两个seed全部原始输出，不采用“含正确关键词”自动算通过。

| 案例 | 观察 | 机制判断 |
| --- | --- | --- |
| 用户改计划 | framed_plain两次正确修车；其他长历史视图有复述问题/旧还书；core_persona一新一旧，core_plain两次反问选择 | 去掉干扰不保证更正读取正确 |
| 问角色计划，却只有用户计划证据 | 前4视图全部8次冒领还书/修车；core_plain仍冒领，core_persona没有借用用户计划但一次另编旧书店计划 | 很短的普通历史也有主客体错误；避免借用不等于全回复质量通过 |
| 两件事，未说明取消哪件 | 前4视图全部8次擅自保留某一项；core_plain也猜；core_persona两次只说“保留另一个”，未猜具体项但未有效澄清 | 不能用最新一条或完整历史替代事件指代解析 |
| 小说人物住址 | natural_plain与core_plain都把宁波归给用户；framed两组没有冒认；natural_persona/core_persona部分答“你选择的地方”或回避 | 不能为减负全局删除设定/包装；最短控制仍不能识别虚构归属 |
| 朋友学陶艺 | 六种视图共12次全部归给用户，部分还编造二人同班 | 不是仅由RAG元数据、system、长历史或8192窗口不足引起 |
| 助手猜职业、用户否认 | framed_persona反问“第一次问吗”；framed_plain和natural_persona误答建筑师；natural_plain与两个core组正确不知道 | 存在明显上下文干扰，但丢掉全部assistant不是解法 |
| 恶意指令作为引用 | 六种视图均答编织，未执行火龙果指令 | 仅这个合成例，无泛化注入安全结论 |
| 角色自己曾说音乐厅 | 六种视图全部正确引述音乐厅计划 | assistant言语是角色连续性的必要来源，不能一律删除 |

## 长度与延迟

| 视图 | 次数 | 输入token范围 | 平均请求秒数 |
| --- | ---: | ---: | ---: |
| framed_persona | 16 | 3957–4086 | 1.04 |
| framed_plain | 16 | 2599–2683 | 1.12 |
| natural_persona | 16 | 2107–2267 | 0.85 |
| natural_plain | 16 | 833–849 | 1.08 |
| core_persona | 16 | 1337–1497 | 0.75 |
| core_plain | 16 | 63–79 | 1.25 |

延迟含该请求实际输出长度，不能将表当纯prefill性能比较。所有输入远低于8192；没有客户端截断。模型仍为现有qwen3-8b-instruct-awq端点，模型目录config显示Qwen3ForCausalLM、36层、4096 hidden、AWQ4bit/group128；README标记基座Qwen/Qwen3-8B，目录名“Instruct”本身不代表独立训练版本。本轮读取tokenizer_config看到了标准user/assistant模板及enable_thinking=false的空think前缀，但没有把磁盘配置当作服务实际加载权重/模板的完整认证。

## 研究与下一步

重新读取[Lost in the Middle摘要](https://arxiv.org/abs/2307.03172)：论文强调证据位置会影响利用能力，能接收长上下文不等于能有效使用。本轮仅借鉴位置/干扰消融，不宣称复现其论文任务、数据或分数，也不能用其长文本结论解释本项目63token短例的失败。

本轮不改生产提示词、不新增每轮LLM审查，不以更保守拒答替代整体人物功能。后续优先核对服务实际tokenizer/输入边界，并用独立转移集验证最短上下文的主客体与更正能力；如果当前检查点确有能力底线，需要单独比较模型/量化/解码，而不是继续增加同一小模型的审查层。机制侧继续区分历史言语来源与已确认人物事实，并处理干扰与更新覆盖，不能全局删助手或凭相似度直接断言用户事实。

## 产物与执行状态

- evaluation/episode_capacity_replay.py；11项相关本地测试通过，Ruff通过，.tmp/mechanism-r90-local.xml。
- .tmp/mechanism-r90-{replays.jsonl,manifest.json}：64次。
- .tmp/mechanism-r90-core-{replays.jsonl,manifest.json}：32次。
- 远端evaluations/mechanism-r90-source、mechanism-r90-capacity-auth、mechanism-r90-core保留；均REMOTE_EXIT=0。
- 首次从错误位置加载.env，首次HTTP返回401，0次成功生成。正确配置位于backend/.env；仅加载到隔离进程，未打印密钥、未修改配置。失败目录mechanism-r90-capacity保留，不计入96次成功调用。
- 未下载/启动新模型，没有重启、迁移或发布生产。
