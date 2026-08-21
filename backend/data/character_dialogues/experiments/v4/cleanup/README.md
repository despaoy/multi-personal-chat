# 月社妃 V4 清洁候选

本目录保存针对正式训练长度审计生成的清理策略、候选、辅助数据和晋升记录。候选已于 2026-08-21 晋升到活动 `../train.jsonl`，但整个 R0 仍处于重新冻结阶段。

## 处理结果

- 原训练集：948 条。
- 清洁候选：926 条，1961 个实际监督 assistant 目标。
- 技术辅助数据：22 条，完整保存在 `candidate/technical_auxiliary.jsonl`。
- 场景元数据修复：8 条，仅修复乱码的 `metadata.scene`，不改对话正文。
- 精确分词验证：Qwen3 chat template 下最长 1165 tokens；候选训练长度设为 1280 后无截断。

## 文件说明

- `length_cleanup_policy.json`：排除 ID、截断损伤 ID、场景修复值和目标长度。
- `candidate/train.jsonl`：待复查的正式训练候选。
- `candidate/technical_auxiliary.jsonl`：从正式人物训练中移出的通用代码能力样本。
- `candidate/manifest.json`：数量、来源、监督目标、分词审计和后续阻塞项。
- `promotion_approval.json`：项目负责人授权的机械清理晋升记录；不声称重新逐条人工审核 926 条数据。

当前正式训练集已经与 `candidate/train.jsonl` 一致。Gold v3 与 prompt policy 契约重新绑定前，不得启动正式 R1 实验。
