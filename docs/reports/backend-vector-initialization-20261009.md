# 角色向量模型初始化状态修复（2026-10-09）

## 问题和修改

`LocalMeanPoolingEmbeddingProvider._load` 在 eval 和 CPU 设备准备完成之前写入 self._model。初始化中途失败后，下一请求仅检查 self._model 非空，会直接复用未完成初始化的模型。

改为使用局部变量完成 tokenizer、模型加载、eval、设备准备，全部成功后再保存实例。沿用现有锁；没有新增状态标志、重试、异常吞噬或降级路径。生产代码净增加 1 行。

## 验证

服务器定向运行 test_vector_initialization.py 和 test_embedding_gradient_isolation.py：5 passed，2.36 秒。

覆盖 tokenizer 加载、模型加载、eval、设备准备各阶段失败时保留原始异常且不发布实例；本次调用不重试，下一独立请求重新完整初始化，成功后复用实例。保留 local_files_only=True，确认初始化不改变调用线程的梯度状态。

使用明确模型替身模拟故障，不声称真实权重或 GPU 推理验证。无全量回归、无下载、无付费模型调用。未改动当前模型配置和索引。

部署健康、代码规范及版本同步记录保存在服务器本阶段备份目录。不提交运行数据、密钥或私有原文。
