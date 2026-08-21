# KISAKI V4 说话者契约

## 为什么仍使用 user / assistant

`role` 表示模型对话协议，不表示剧情人物：

- `assistant`：始终是被训练的月社妃。
- `user`：当前向月社妃说话的一方，可以是原作人物，也可以是普通用户。

Qwen3 的聊天模板需要标准 `system/user/assistant` 角色。把 `user` 改成“琉璃”或“夜子”会破坏模板兼容性，也会把协议角色和人物身份混在一起。

## 记录级字段

原作具名人物使用：

```json
{
  "interlocutor_kind": "canonical_character",
  "interlocutor_label": "琉璃"
}
```

无法证明具体身份的构造对话使用：

```json
{
  "interlocutor_kind": "generic_user"
}
```

文本中出现“夜子”或“理央”只说明谈论了该人物，不能据此推断发言者身份。

## 消息级字段

一段对话中有多个发言者时，在各条 `user` 消息上标注：

```json
{"role": "user", "content": "……", "speaker_label": "夜子"}
```

训练器按以下优先级解析：

1. user 消息自己的 `speaker_label`。
2. 记录级 `interlocutor_label`。
3. 没有可靠身份时不注入姓名。

`generic_user` 禁止附带具名说话者；`canonical_character` 必须提供可靠标签。消息级标签只能出现在 user 消息上。

## 安全边界

说话者标签属于不可信上下文，只能放入 `<speaker_label trust="untrusted">`，不得拼接到 system prompt。训练输出仍只包含标准 `role/content`，说话者元数据不参与 assistant loss。

当前 522 条原作训练记录使用具名人物标签；404 条已审核构造记录使用 `generic_user`。107 条原作多轮记录的历史回合已通过保存的原文行和审核记录核验，未发现第三人物混入。
