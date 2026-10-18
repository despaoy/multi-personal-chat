# llama.cpp 消息协议精简（2026-10-19）

基线：`e361a8d0d5e16a11185f7c740b17c280535e1085`。服务器先修改、验证，随后同步本地。

## 调用链与问题

`api/generate.py` 通用模型分支、交互评审脚本及作品检索评估会把编译后的消息拆为末条问题和前序历史，传入 `ModelManager.async_generate`。其中系统消息包含角色约束、证据和授权范围，不能按只包含用户/助手的聊天记录处理。

`LlamaCppProvider.async_generate` 原先将历史手工拼接为 `User:` / `Assistant:` 文本，忽略系统消息，也给缺失的角色或正文补默认值。它调用原生 `/completion`，无法让模型服务应用实际聊天模板。修改前服务器隔离复现确认系统消息未进入请求。

## 修改

- 使用现有地址处理器调用 `/v1/chat/completions`，原样保留全部消息的角色、正文、顺序和末条问题，由服务应用模型自带模板。历史必需字段直接读取，不补造数据。
- 使用官方聊天协议的 `max_tokens`；温度、top-p、超时、共享客户端和配置刷新行为保持一致。单模型服务采用已加载模型，不发送显示名称作为模型别名。
- 复用已有 `completed_chat_content` 处理完成状态与有效正文，删除重复解析。
- 请求 `verbose: true` 和 `response_fields: ["truncated"]`，仅取得所需截断标记；不要求返回原始提示词、生成设置或重复正文。截断或缺失必要标记会报错，没有旧接口重试或降级。

生产代码净减少 **13 行**。没有新增模块、配置项、重试、语义规则或分类器；其他模型的方法 AST 保持一致，当前 DeepSeek 路径没有改变。

## 验证

- **33 项定向测试首次全部通过**：llama.cpp 及共享测试夹具涉及的 Ollama 配置刷新、输出预算、失败传播、资源释放和完成状态。新增 5 个针对完整历史、缺失历史字段和截断元数据的契约案例。没有运行全部回归。
- **12 个服务器前后对照案例**：真实 SQLite 配置和真实 httpx 客户端，HTTP 传输使用明确的受控响应。完整的虚构事实包括人物、时间、金额、未付款、天气例外和书面改期授权；13 条历史加当前问题完整送达。旧实现丢弃系统消息，新实现保留。缺失字段原先发送补造内容，现于取得客户端前失败。
- 无历史成功路径、空正文、未完成、输出限额、上下文截断、缺失元数据/正文、错误类型和 HTTP 故障均检查；每次最多一次请求，输入不变，客户端释放。保留原业务断言，仅依协议调整 token 字段名称和响应夹具。
- 差异检查、限定静态检查、仓库完整性检查通过。部署后后端就绪和前端登录均为 200，受保护接口仍返回 401；配置指纹和其他服务进程保持一致。

未调用付费模型或真实 llama.cpp 权重，未验证其实际回答质量；本阶段证明消息与响应协议，不声称模型质量保证。服务端需支持这里使用的官方聊天接口及截断元数据；缺少支持时会明确失败。

## 官方协议依据

- [llama.cpp 服务文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)：聊天模板、完成状态和响应字段筛选。
- [聊天参数解析](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-common.cpp)：解析完整 messages、应用模板并传递原生参数。
- [参数 schema](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-schema.cpp)：`verbose`、`response_fields`、`max_tokens`。
- [响应序列化](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-task.cpp)：聊天响应的 `__verbose` 采用已筛选的原生响应，保留 `truncated`。
- [请求处理](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)：聊天接口与实际加载模型的关联。

整体优化目标继续；本次完成一个可独立验证的协议阶段。
