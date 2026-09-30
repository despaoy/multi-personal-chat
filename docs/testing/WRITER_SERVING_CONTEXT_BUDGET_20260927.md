# r146：按完整请求与服务窗口编排记忆写入

## 机制

writer 改为可使用配置的服务窗口估计完整序列化请求成本：系统提示、当前消息、完整历史轮次、候选记忆，加 768 输出预留和 512 安全余量。复用主回答的同一无 tokenizer 估计函数，提取至 inference/context_budget.py，主回答算法不变。不得宣称估计等于真实 tokenizer。

MemoryLlmConfig.from_env 默认继承 VLLM_MAX_MODEL_LEN（缺省8192），外部模型可显式配置 MEMORY_LLM_CONTEXT_WINDOW_TOKENS；设0保留旧字符预算，直接构造 config 也保持旧默认兼容。已更新 .env.example。窗口模式不再受当前2000字和历史2000字独立门槛限制，但仍保留完整最近4消息轮次；整体放不下仍skipped/input_budget，不截断。没有新模型层、没有人物prompt修改。

## 本地验证

4项新测试覆盖完整长消息、长历史后的短消息、序列化总成本临界点、服务配置/旧模式。扩大回归2489 passed、1 skipped、1784 deselected、4依赖警告，155.30秒，`.tmp/r146-wide.xml`。8项聚焦与总数重叠不累计。Ruff/diff通过。

## 同一失败场景真实重跑

显式8192 writer窗口，无LoRA、隔离SQLite、真实API，2轮HTTP200、7次DeepSeek（1semantic2policy1answer2writer1selection）、0本地生成。`.tmp/r146-manifest.json`、`.tmp/r146-traces.jsonl`。

- 2281字完整入writer；专业与末尾偏好两条实际saved，原话完整保留。
- 下一轮确认有专业记录；writer收到2281字历史及短当前问题，正常NOOP，不再因固定历史字数跳过。
- 第一次writer估计输入3820，含预留5100；API实际prompt_tokens2496。第二次估计3964，含预留5244；实际prompt_tokens2625。两次均在8192基线内，不依赖强模型理论大窗口。
- source长query错误、整段source预算遗漏仍存在，只是当前结构化记录可正确回答，不能说来源链路已修好。

## 独立迁移：长陈述→短更正→冷历史追问

3轮HTTP200、10次DeepSeek（3policy3answer2writer2selection）、0本地生成；问题阶段prepared.history为空。`.tmp/r146-update-{manifest.json,traces.jsonl}`。

- 工作地点首写自然博物馆，随后天文馆记录active，supersedes_memory_id指向旧记录，旧记录superseded。未靠近期历史回答；最终答对天文馆。
- 但读视图把证据“更正一下，我现在在天文馆工作，不在原来的单位了。”降为observation，并注入“时效未核实，不代表当前状态”。回答随之追加“没有其他证据可以核实”。这是读取机制过于保守的新定位，不是写入失败或必须改提示词；不能报告整体人物体验通过。

## 尚未完成

完整更正句的主体/谓词/否定与时间作用域解析；source长query及原话整段预算；source-only删除；异步保存确认；动态语义独立窗口和消融；RAG充分性、并发与全验收均继续。旧配置切换已在示例说明，不发布生产。远端会话已关闭，持续目标active。
