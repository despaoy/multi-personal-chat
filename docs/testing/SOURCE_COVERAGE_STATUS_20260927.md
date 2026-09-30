# r145：来源覆盖不等于字段不存在

根因：attach_sources 只在原话真正入模时将缺失字段从 False 改为 None；预算剔除/检索失败时反而沿用 False。重复 attach 空结果还可能留下上一次非共享来源文本。

修复：CompiledCharacterContext 独立记录 memory_source_status；budget_omitted/retrieval_error 撤销不存在证明，不改变已存在字段，不晋升原话为事实。刷新时清空旧来源包；保存状态解释未能核对原话，不声称没有保存。不新增模型、不改 prompt。

4项先失败测试、1项兼容测试；33聚焦通过；扩大回归2485 passed、1 skipped、1784 deselected、4依赖警告，156.10秒，`.tmp/r145-wide.xml`。Ruff/diff通过。聚焦不累计。

真实隔离服务器2轮HTTP200、4次DeepSeek（1semantic2policy1answer），0writer、0本地生成。来源完整保存、writer仍input_budget；下轮source预算剔除时字段为None，明确不能确认而不是报无记录。`.tmp/r145-{manifest.json,traces.jsonl}`。仅修复状态传播，不算长文记忆通过，继续r146实际窗口预算。生产未改，目标active。
