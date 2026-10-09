# 纠正检索开关统一解析（2026-10-09）

普通生成入口与 GroundedAnswerService 分别使用宽松布尔解析，无效字符串会静默禁用纠正流程。两处现复用 infra.environment.read_bool，并移除 GroundedAnswerService 的重复 _env_flag。普通生成在索引检查前读取 CORRECTIVE_RAG_ENABLED，配置错误不触发索引工作。

保留普通生成默认关闭、GroundedAnswerService 默认开启，以及显式 corrective_enabled 优先于环境变量的既有规则，不更改合法配置行为。角色等未使用普通知识库纠正的路径不额外校验这个无关开关。

服务器 11 项定向测试通过（3.53 秒）：错误/空开关在索引检查前报错，带引用服务默认/启用/禁用/非法值及显式覆盖保持约定，两项实际纠正流程与来源读取检查通过。四条警告来自既有 jieba。无全量回归或付费模型。
