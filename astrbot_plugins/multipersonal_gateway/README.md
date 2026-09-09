# multipersonal_gateway

> 状态：薄网关插件。平台登录与账号凭证由 AstrBot 管理，RAG、LoRA、推理、历史和策略仍由 FastAPI 负责。

AstrBot 插件，用于 MultiPersonal Chat System。它将 AstrBot 保持为多平台网关，并将归一化后的文本消息转发至 FastAPI：

`POST /api/integrations/astrbot/messages`

本版本需与支持发送回执的后端同步升级，并先执行后端数据库迁移。插件在平台发送成功后
调用 `POST /api/integrations/astrbot/delivery`；发送失败保留后端已生成结果，下次收到同一
事件时可复用。暂时失败不会被插件标记为已完成。详细状态与故障边界见后端 README 的
“平台发送与重试”。

环境变量：

- `MULTIPERSONAL_BACKEND_URL`（旧名 `QQCHAT_BACKEND_URL` 仍兼容）：FastAPI 基础 URL，默认 `http://127.0.0.1:8000`
- `ASTRBOT_INTEGRATION_TOKEN`：共享令牌，必须与后端配置一致
- `MULTIPERSONAL_TRIGGER_PREFIXES`（旧名 `QQCHAT_TRIGGER_PREFIXES` 仍兼容）：群触发前缀，默认 `/ai,/chat,@bot`
- `MULTIPERSONAL_REPLY_GROUP_ALL`（旧名 `QQCHAT_REPLY_GROUP_ALL` 仍兼容）：设为 `true` 时转发所有群消息
- `MULTIPERSONAL_BACKEND_TIMEOUT`（旧名 `QQCHAT_BACKEND_TIMEOUT` 仍兼容）：HTTP 等待秒数，默认 `210`，最小 `16`。插件在请求中传递总处理预算，取 `180` 与 HTTP 等待时间减 `15` 的较小值；后端再以 `MODEL_INFERENCE_TIMEOUT` 限制预算。
- `MULTIPERSONAL_DEDUP_TTL`（旧名 `QQCHAT_DEDUP_TTL` 仍兼容）：内存事件去重 TTL 秒数，默认 `300`
- `MULTIPERSONAL_QQ_ADAPTER`（旧名 `QQCHAT_QQ_ADAPTER` 仍兼容）：QQ 事件的适配器标签，默认 `napcat`
- `MULTIPERSONAL_WECHAT_ADAPTER`（旧名 `QQCHAT_WECHAT_ADAPTER` 仍兼容）：个人微信事件的适配器标签，默认 `gewechat`；推荐值：`gewechat`、`wechatpadpro`、`other`

个人微信说明：

- 个人微信的登录和凭证保留在 AstrBot 和所选适配器内部。
- 本插件仅归一化 AstrBot 事件。若 AstrBot 上报的平台名包含 `wechat` 或 `gewechat`，事件以 `platform=wechat_personal` 转发。
- 生产部署应优先使用企业微信或公众号；个人微信适配器最好视为实验性。

本插件有意不实现 RAG、LoRA 或模型推理。这些能力保留在 MultiPersonal Chat System 后端。
