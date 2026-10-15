# 删除无当前消费者的旧索引持久化与磁盘缓存

日期：2026-10-16。基线：`b9bf76e597e329388afcf00c92a64f4957fdde4d`。

## 调用审计与精简

当前索引构建通过 `CharacterKnowledgeIndexBuilder` 和 `write_vector_artifacts` 产生 documents.jsonl、vectors.npy、manifest.json，由 `multiscale_rag.runtime` 严格加载。`DomainIndex` 只作为内存检索结构使用。

本地全量引用审计及服务器 tracked backend/scripts/src 预检均确认：`EmbeddingCache`、`embedding_cache_key`、`build_domain_index` 除定义和包导出外无当前消费者；DomainIndex 只有实际 runtime 和旧 builder 两个构造位置，实际 runtime 从未调用旧磁盘 load/write。

删除旧 faiss.bin/index_manifest.json 读写、路径属性、旧 builder、独立 NPZ EmbeddingCache 与对应三个导出。移除两个“加载失败后清空并继续”的旧异常分支。DomainIndex 保留实际向量、稀疏、实体检索、统计及健康接口；唯一真实构造不再传入虚构磁盘目录。生产代码净减少约 293 行，没有新增抽象或替代链路。

保留真实模型路径解析、指纹、加载锁、编码批次与向量校验，以及记忆服务自身的向量缓存和查询编码缓存。未删除数据、索引产物或历史 archive，未改用户功能、配置或外部 API。

## 验证

14 项直接相关检查最终通过。首次 13 项通过，1 个旧 runtime 用例因缺少模型路径前提停止；进一步核对还发现其整对象假编码器带有与实际产物不符的模型标识。该单测现使用真实 provider 保留正确身份，仅在查询编码边界显式提供合成输出，并明确禁用此单测不评估的交叉编码器。1098 文档、六类数量、结果、域、路由与信任属性等原断言全部保留。

仅重测这一个未完成用例，1 项通过（10.09 秒），没有重跑已经通过的 13 项。相关检查覆盖两种 BM25 的正常/失败与批次发布、记忆语义检索及向量缓存复用、实际三分区产物→默认 runtime→检索/来源读取→回答缓存，以及向量内积分数和实际 provider 编码契约。没有新增只检查实现形状的测试，没有全量回归。

删除旧解析代码后，仅用于类型的文档导入移入 TYPE_CHECKING；最终语法、补丁和增量静态检查无新增问题。初次日志和单项复测日志均保留，未修改原始失败记录。第三方 jieba 的既有警告未在本阶段处理。

模型输出、向量和排序为明确的测试控制，不是实际模型质量评测，没有真实模型调用或费用。沿用服务器先改与验证、仅重启后端、健康及配置/其他服务核对、逐文件 SHA-256 同步本地、阶段提交 GitHub 的交付流程。DeepSeek 认证问题仍等待配置更新，未再次请求。
