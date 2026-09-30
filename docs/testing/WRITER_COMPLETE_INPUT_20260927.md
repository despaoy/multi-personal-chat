# r144：写入输入完整性与长文本链路缺陷

## 复现及修改

4 项先失败测试证明：writer 当前消息按 max_input_chars 截前缀；历史只留每条前 500 字；四条消息边界可拆开用户限定和助手回答。即使强模型也看不到被截掉的语义。

复用完整轮次窗口，允许配置消息数与总字数；writer 历史保留最多 4 条、总计 2000 字的完整后缀轮次。取消服务入口提前四条切片。排队保留历史，worker 捕获完整来源后才验证模型输入，避免预算错误在来源捕获前丢任务。当前消息超配置预算不截断，返回 skipped/input_budget，0 persisted；历史超预算同样处理。不增加模型调用或改人物提示词。

这只是防止不完整输入造成误写，不是长文本提取能力已完成。当前仍为固定字数预算；预算检查还在候选检索之后，存在可避免的检索开销。

## 本地证据

6 新测试包括实际 SQLite：当前消息/历史超预算均保留完整 source、无 claim、无 writer 调用，返回准确跳过原因。扩大回归 2480 passed、1 skipped、1784 deselected、4 依赖警告，153.79 秒，`.tmp/r144-wide.xml`。Ruff 与 diff check 通过。首次受限聚焦运行 29 passed/2 tmp_path 权限错误，不算全通过；之后获准扩大回归覆盖。

## 真实服务器两批

无 LoRA、隔离 SQLite、真实 API、所有启用内部 LLM cloud 注入，生产未改。

1. 5 轮 HTTP 200、15 次 DeepSeek：2 semantic_review、5 policy、5 answer、2 writer、1 selection，0 本地生成。`.tmp/r144-manifest.json` / `.tmp/r144-traces.jsonl`。
   - 虚构长文先被已有 skipped_fiction_or_note 拦截，没有直接验证当前长消息预算分支。随后短追问因历史超预算被 writer 跳过。
   - 679 字历史末尾的“不是我的真实住址”确实完整进入真实 writer recent_history（先前会丢掉）。真实专业写入，下一轮专业正确、未把小说地点当真实住址。
   - 模型对重复压力文本产生不必要的心理动机推测，不能宣称人物体验通过；这不是用例自然度评价，也不靠实体特判修复。
2. 迁移 2 轮 HTTP 200、4 次 DeepSeek：1 semantic_review、2 policy、1 answer、0 writer，0 本地生成。`.tmp/r144-transfer-manifest.json` / `.tmp/r144-transfer-traces.jsonl`。
   - 2281 字可准入陈述完整 source 留存；writer 确实 skipped/input_budget，无结构化 claim，未调用模型。
   - 首轮说“记住了”，下轮确定性答“当前可用的长期记忆里，没有你的专业记录”。原话存在但被 source whole_source_budget 剔除。这是端到端体验失败，不能视为本轮全过。
   - 首轮 source recall 因长 query 返回 retrieval_error/ValueError；后续 source 检索命中但预算剔除。短追问 writer 也被前一长历史挡住。

## 下一步优先级

1. 完整长文写入的实际模型预算编排，区分当前消息独立性与必要历史；不能靠全量拒绝或盲目加大长度收尾。
2. source 命中但整段预算剔除、仅来源保存与结构化字段存在性的契约；不能把“未入模/未结构化”混作“用户没说过”。
3. 普通显式记住的保存确认与实际异步结果不一致；避免仅提示词约束或又加复核模型。
4. 语义状态独立 6 条窗口、source-only 删除、RAG充分性、动态消融及总验收继续未完成。

远端会话已关闭，目标保持 active。
