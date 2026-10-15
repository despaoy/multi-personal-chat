# 移除旧依赖字段自动转换

日期：2026-10-16。基线：`57ec147de0b682472a2202d2fcef4a848694cb2f`。

全仓核对 parse_dependencies 的真实生产者、调用方和读取方：主聊天调用 plan_retrieval_views，唯一生产输入为本次语义审核 JSON，当前提示词明确要求规范五键及 unresolved_source。TurnDependencies 只在 parser 中构造，没有数据库或文件旧计划的反序列化入口；离线 evaluation/scripts 也没有旧计划生产者。规范输出下游直接使用 unresolved_source。唯一直接依赖旧 unknown 转换的是两项历史兼容测试。

服务器安全预检使用当前提示词格式的完整 query、逐字搜索片段和 public_partitions，只将依赖输出字段换成旧 unknown；空列表和含索引两种旧输出都被自动接纳为 applied。现删除这段自动转换，共移除 10 行生产代码。旧字段输出现在由现有 parser 拒绝，再由现有 planner 报 QueryPlanReviewError(status=invalid) 并保留 ValueError 原因；不添加模型补救、字段猜测或无检索生成。规范字段的完整覆盖、严格索引、重复及重叠语义检查均保留，没有持久化迁移或删除历史资料。

服务器 10 项现有定向检查通过（0.26 秒）：旧 unknown 空/非空列表均拒绝；规范 control/unresolved_source 重叠 2 项、完整私人依赖 1 项、同句私人及公共依赖 1 项、真实 planner 的缺字段及旧字段错误传播 3 项，以及完整规范问题到审核的健康路径 1 项。修改原有兼容预期并增强 planner 的状态及原始异常断言，没有新建测试文件或放宽有效检查，没有重复全量回归。

语法、增量静态与差异检查通过后只重启后端，核实就绪、原配置和其他服务，再按 SHA 同步本地、服务器 Git 与 GitHub。审核全部使用明确的合成返回，不调用真实模型，不提交密钥、运行数据或私有对话。真实 DeepSeek 输出质量未在本阶段验证。
