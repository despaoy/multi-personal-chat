# 嵌入模型路径统一（2026-10-09）

## 问题与修改

通用 VectorDatabase 与共享 embedding provider 各自维护目录查找。显式 EMBEDDING_MODEL_PATH 无效时会继续查找；共享解析器找不到指定模型缓存时会尝试默认模型；通用向量库还包含不同中文模型的备用目录。查询模型与索引模型身份因此可能不一致。

通用向量库现在直接使用 retrieval_core.embedding.resolve_local_model_path，删除重复查找器和没有调用方的 EMBEDDING_MODELS 别名表。显式路径要求是包含 config.json 的目录，无效时抛出 EmbeddingModelError 并提示修复配置，不进入缓存查找或远程加载。

HF 缓存只匹配请求的模型身份，正确处理带组织名称的模型 ID。不再用默认模型替代其他模型。默认模型的 backend 目录、HF snapshots 及已配置 MULTIPERSONAL_LAB_ROOT / QQCHAT_LAB_ROOT 目录仍可使用；只有明确开启远程加载才返回请求的模型名。移除 ModelScope 中另一模型的隐式替代路径，不删除任何模型文件。

## 验证

12 项定向测试通过（0.42 秒），覆盖显式错误路径、有效路径优先级、禁止跨模型缓存替代、精确组织名缓存、远程显式启用、两种实验目录配置，以及 VectorDatabase 和角色 provider 的实际调用入口。

首轮因测试引用了不存在的 VectorDB 类而在收集阶段失败，纠正为实际 VectorDatabase 后执行通过，没有修改业务断言。

另用服务器运行进程的实际环境进行隔离检查：共享 provider 与角色 provider 解析为同一存在的模型目录，通过。环境值及诊断留在服务器，未输出凭据。

没有重跑全量回归、下载模型或调用付费模型。路径测试使用合成目录和模型构造替身；实际环境检查验证路径解析，不声称重新验证模型权重或推理质量。本阶段没有调整 GPU 行为和已有索引。

部署及同版本同步结果存于服务器本阶段备份目录。
