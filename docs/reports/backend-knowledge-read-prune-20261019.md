# 2026-10-19 删除旧知识库读取入口

基线：`bbbda3ed3cef23e0c5af4b21fc24c34ecd3e0201`。

## 调用链与问题

通用知识库的索引指纹、元数据校验及重建都调用 `iter_chunks_with_document`。SQLite 使用独立只读事务；PostgreSQL 使用 `iter_chunk_document_batches` 提供快照批次，异步行迭代与同步适配器分别消费该批次接口。读取完成、提前关闭、异常及适配器停机由该链路释放资源。

旧的全量读取、普通分页迭代和单页文档连接查询已经退出这条链路。仓库检索及 Python 调用引用核对发现，除旧方法之间相互调用外，没有业务、脚本或测试调用方。仅一条历史测试仍检查旧辅助方法必须存在，其描述与现用同步适配器不符。

## 修改

删除七个内部方法：

| 类 | 删除方法 |
| --- | --- |
| `SQLiteDB` | `get_all_knowledge_chunks`、`iter_all_knowledge_chunks` |
| `PgDatabase` | `get_all_knowledge_chunks`、`iter_all_knowledge_chunks`、`get_chunks_with_document` |
| `SyncPgAdapter` | `get_all_knowledge_chunks`、`iter_all_knowledge_chunks` |

两个生产文件净减少 **132 行**。保留方法的 AST 全部相同，未改变现用查询、事务、索引生成或资源释放代码；未修改数据库结构及历史数据。现有上传、文档读取、检索和索引入口继续使用原有实现。

将唯一过期的方法存在性断言更新为实际被同步适配器调用的 `iter_chunk_document_batches`，修正文档注释；断言总数和其他断言保持不变。没有新增测试案例、校验、配置、包装、重试或回退。

## 验证

- 在服务器运行相关知识库读取、索引、扫描、搜索、文档和驱动契约检查：**130 项首轮全部通过，零跳过、零失败**。
- 现有快照测试检查跨页读取期间的更新、插入与删除，以及提前关闭、异步页读取失败、同步适配器停机时的资源释放。
- 现有索引测试检查跨进程修订、元数据变更、原文完整性、索引发布及失效后的明确失败；搜索错误仍返回失败，正常空结果不启动其他检索器。
- 全仓库受版本控制的 Python 文件中，七个已删除方法的定义及调用引用均已消除；现用读取方法继续存在。
- 未用导入、重复定义、未定义引用检查通过；仓库完整性与差异检查通过。
- 后端加载新代码后，就绪接口和前端登录均为 200；受保护接口为 401 并保留安全响应头。配置和前端进程保持不变。

本阶段使用真实 SQLite 及现有受控异步后端的桥接测试，没有新增真实 PostgreSQL 数据库实验，也没有付费模型调用。现用 PostgreSQL 读取实现保持原样；不把结构校验或受控后端结果标记为真实 PostgreSQL 执行。未重复运行完整后端回归，前一整体验证范围见[报告](backend-post-owner-overall-20261019.md)。

运行证据保存在服务器 `/home/boot/lhm/multipersonal-runtime/backups/backend-chain-20261001/knowledge-read-prune-20261019/`，包含原文件、哈希、移除证明、定向结果及部署状态，不纳入 Git。
