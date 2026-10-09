# 记忆持久化默认值清理（2026-10-09）

## 调用链与问题

后台调度器构造必含原始 message 的 `_MemoryJob`，解析后的提案交给 `_persist_proposal`，后者调用返回整数删除数量的仓储接口。原实现再次对 message 使用空字符串默认值、对删除数量进行 `int(deleted or 0)` 转换；需要写入的提案缺失 memory 时返回 no_change。这些处理掩盖内部契约错误。

## 修改

- 直接读取 job.message，保留未来生效时间判断所需的完整原文。
- 直接比较仓储删除数量，不将 None 默认为零，也不接受字符串数字作为内部返回值。
- 非 NOOP、非立即 ERASE 的写入提案缺少 memory 时抛出 ValueError，不伪装为无需修改。
- NOOP、真实零删除数量及仓储明确 persisted=False 的正常无变更仍保留。

生产代码没有增加行数或抽象层。补齐两个旧测试替身的 message 字段，以及既有持久化测试模型提案的 attributed_to 字段，保持原有断言不变。

## 定向验证

服务器运行 test_persistence_cleanup.py、test_deferred_memory_mutation.py、test_future_withdrawal.py、test_future_erasure_source.py、test_memory_persistence_contract.py：65 passed，1.71 秒。

覆盖缺少写入内容的六种操作、非法删除返回值、正常删除与无变更、未来撤回/删除不能提前执行，以及失败/成功持久化回执。测试使用完整合成输入；部分测试使用明确的仓储/模型替身，未来生效测试使用临时 SQLite。

无全量回归，无付费模型调用。没有声称验证真实模型质量或所有持久化并发场景。内部整数契约仍由仓储负责，本阶段没有新增重复的全类型校验。

部署与同版本同步检查保存在服务器本阶段备份目录。私有原文、密钥和运行数据不提交。
