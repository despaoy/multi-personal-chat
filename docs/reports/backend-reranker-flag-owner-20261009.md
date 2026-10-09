# 重排开关统一解析与死代码清理（2026-10-09）

RAGHelper、PipelineReranker、角色多尺度服务各自判断重排开关，RerankConfig 另有重复下载开关解析；拼写错误与空字符串都被默认为 false。统一复用 infra.environment.read_bool，非法值明确报出配置名。保留默认关闭、角色/通用独立开关及显式注入行为。

删除两个重复 _env_flag 实现、无调用方的 rerank_documents 便捷入口和无读取方的 RAGHelper.rerank_top_k。全仓跟踪文件引用检查仅发现这两个定义；实际服务继续使用 get_reranker 和 PipelineReranker，不删除重排能力。

服务器 19 项定向测试通过（2.50 秒）：四个配置消费入口各验证启用、禁用、拼写错误和空值；错误/关闭时不初始化编码器。既有角色独立开关和已启用重排初始化失败检查继续通过。无全量回归、无付费模型。仅清理明确的重复配置逻辑，本阶段未改变缓存 TTL 或排序算法。
