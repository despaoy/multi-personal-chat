# 月社妃 V4 人工审核与重训练状态

## 当前状态

- 正式训练：Game Train 复审已关闭，等待当前环境正式门禁复验
- V4 train：当前 948 条（522 原作 + 150 既有构造 + 4 条 DeepSeek round06 五轮会话 + 272 条 Codex 自动化批次五轮会话）
- V4 validation：70 条已冻结
- Gold v2.1：150 条，已批准为 `development_only`
- Gold v3：150 条最终盲测，已审核并冻结

## 为什么需要 V4

旧正式训练集包含 111 条 `llm_v3_deepseek` 样本，存在元叙事过载、句式模板化和角色语气偏移。旧训练器对多轮数据只监督最后一个 assistant 回复，也会浪费早期回合。V4 先由项目负责人检查人物画像、原作提取、构造数据、验证集与 Gold，再重新进行单变量 PEFT 消融。

## 人工审核入口

人物画像、system prompt、构造数据、validation、Gold v2.1、Gold v3 和 Game Train 上下文质量均已有批准记录。最终 Game Train 审核将 576 条收敛为 522 条，其中 107 条补充上下文并仅监督最后一个 assistant 回合；Gold 内容未修改，污染复审为 `clean`。

## 训练门禁

`scripts/validate_kisaki_v4_training_gate.py` 会检查：

- 必需审核分类是否全部明确通过。
- V4 train/validation 是否冻结并具有哈希。
- Gold v3 是否冻结且为 150 条。
- 可选的服务器剩余空间是否不低于 15GB。

门禁通过前，不生成正式 R1V4 配置，也不启动 GPU 训练。

## 单向执行顺序

```text
canonical train / validation / Gold v3 已冻结
→ build_kisaki_r1v4_configs.py
→ validate_kisaki_v4_training_gate.py
→ run_kisaki_experiment.py
```

历史候选和逐批复审包已在批准晋升后删除，最终决定以 canonical manifest 与 `game_train_context_review_approval.json` 为准。
