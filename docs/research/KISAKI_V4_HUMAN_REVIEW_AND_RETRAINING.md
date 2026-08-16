# 月社妃 V4 人工审核与重训练状态

## 当前状态

- 正式训练：`blocked_pending_game_context_quality_review`
- V4 train：当前 1002 条（576 原作 + 150 既有构造 + 4 条 DeepSeek round06 五轮会话 + 272 条 Codex 自动化批次五轮会话）
- V4 validation：70 条已冻结
- Gold v2.1：150 条，已批准为 `development_only`
- Gold v3：150 条最终盲测，已审核并冻结

## 为什么需要 V4

旧正式训练集包含 111 条 `llm_v3_deepseek` 样本，存在元叙事过载、句式模板化和角色语气偏移。旧训练器对多轮数据只监督最后一个 assistant 回复，也会浪费早期回合。V4 先由项目负责人检查人物画像、原作提取、构造数据、验证集与 Gold，再重新进行单变量 PEFT 消融。

## 人工审核入口

人物画像、system prompt、构造数据、validation、Gold v2.1 和 Gold v3 均已有批准记录。当前需完成 `review_packets/kisaki_v4/10_FINAL_REVIEW/02_GAME_CONTEXT/` 的上下文质量复审并关闭对应 blocker；Gold 内容不得因训练增补而反向修改，增补后只重跑污染审计。

## 训练门禁

`scripts/validate_kisaki_v4_training_gate.py` 会检查：

- 必需审核分类是否全部明确通过。
- V4 train/validation 是否冻结并具有哈希。
- Gold v3 是否冻结且为 150 条。
- 可选的服务器剩余空间是否不低于 15GB。

门禁通过前，不生成正式 R1V4 配置，也不启动 GPU 训练。

## 单向执行顺序

```text
build_kisaki_v4_canonical_draft.py
→ build_kisaki_v4_review_packets.py（新目录）
→ 人工填写当前 game / constructed / validation 决定
→ freeze_kisaki_v4_dataset.py
→ 建立并审核 Gold v3
→ finalize_kisaki_v4_dataset.py
→ build_kisaki_r1v4_configs.py
→ validate_kisaki_v4_training_gate.py
→ run_kisaki_experiment.py
```

候选构建不读取最终批准文件；审核包不允许覆盖已有人工结果。`freeze_kisaki_v4_dataset.py` 只冻结 train/validation，Gold v3 批准后再由 `finalize_kisaki_v4_dataset.py` 将 canonical 状态推进为 `frozen`。
