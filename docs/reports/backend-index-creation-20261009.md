# FAISS 索引创建契约（2026-10-09）

## 问题和修改

`VectorDatabase._create_index` 对未知类型静默创建平面索引，空字符串还会触发默认类型。IVF 的 nprobe 写在 IndexIDMap 包装层，未设置到底层 IndexIVFFlat。

改为只在参数为 None 时选择配置类型；未知或空类型明确抛出带支持类型提示的 ValueError。将 nprobe 设置在底层 IVF 索引，并合并各分支重复的 IndexIDMap 包装。有效索引构造完成后才赋值 self.index，错误类型不会替换原实例。

## 验证

服务器 test_index_creation_contract.py：12 passed，0.42 秒。

使用真实 FAISS：Flat、IVF、HNSW 均创建成功，对 256 个完整合成向量进行必要训练、添加指定 ID、检索，并断言精确向量返回正确 ID；直接读取原生 IVF 的 nprobe/nlist 及 HNSW 参数。另覆盖配置/显式参数的未知、空、大写错误值拒绝且原实例保留，及自动策略三个规模区间。

没有运行全量回归，没有模型调用。没有修改生产索引文件或触发索引重建；本阶段修复作用于新建索引，不声称已调整磁盘上既有 IVF 索引的参数，也未验证全部迁移和并发行为。

部署、语法与增量规范、同版本同步记录保存在服务器本阶段备份目录。合成向量仅存在测试进程中，未提交运行数据。
