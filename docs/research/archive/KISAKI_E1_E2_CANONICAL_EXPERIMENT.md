# 月社妃 KISAKI-E1/E2 历史实验

> 历史归档：不得使用本文命令启动新实验。当前入口为 `../KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md`。

## 研究问题

在相同数据、模型和训练预算下，NEFTune alpha=5.0 是否改善月社妃 LoRA 的角色质量和稳定性，以及它带来多少训练与推理成本？

## 严格对照

| 项目 | KISAKI-E1 | KISAKI-E2 |
|---|---|---|
| 方法 | 标准 LoRA | LoRA + NEFTune |
| NEFTune alpha | 0.0 | 5.0 |
| 训练数据 | canonical train，哈希固定 | 与 E1 相同 |
| 验证数据 | canonical validation，哈希固定 | 与 E1 相同 |
| 基座 | Qwen3-8B-Instruct BF16 | 与 E1 相同 |
| LoRA | r32、alpha64、7 modules | 与 E1 相同 |
| 种子 | pilot 42；正式 42/43/44 | 与 E1 相同 |
| 生成参数 | temperature 0、thinking off | 与 E1 相同 |

机器会检查配置差异；除 neftune_noise_alpha 外出现任何差异都拒绝启动。

## 当前状态

- canonical train：826 条。
- fixed validation：92 条。
- Gold v1：降级为开发集。
- Gold v2：150 条，五类各 30 条；人工审核、文本与 BGE-M3 语义泄漏审计均已通过并正式冻结。
- KISAKI-E1/E2：Seed 42 canonical 真实训练、自动评测、质量门和 AI 辅助盲审人工确认均已完成。
- 揭盲结果为 E1 胜 26、E2 胜 32、平局 62；E2 非平局胜率 55.2%，双侧精确符号检验 p=0.512。
- 当前只能表述为 E2 呈轻微方向性优势但未达到统计显著，不能声称 NEFTune 已被证明优于标准 LoRA。
- 完整结果见 [Seed 42 盲评报告](reviews/KISAKI_SEED42_BLIND_REVIEW_RESULT.md)。

数量和哈希以 canonical_dataset_manifest.json 为准。

## 评测

- 角色套件：persona、factual、multiturn、safety，共 120 条。
- RAG 套件：30 条，单独评测 Recall@K、MRR、结构化引用、忠实度、拒答和延迟。
- 安全自动规则只作诊断；“噗噗”“你终于来了”等角色语句不能算拒绝。
- 正式结论要求 Gold v2 冻结、真实生成无错误、三种子汇总和人工盲评全部完成。

## 服务器流程

    cd /home/szw/lhm2/qqchat-enhanced
    nohup bash scripts/lab-queue-kisaki-e1-e2.sh pilot > /home/szw/lhm2/runtime/logs/kisaki-queue-nohup.log 2>&1 &

检查 seed 42 后，再运行：

    nohup bash scripts/lab-queue-kisaki-e1-e2.sh replicate >> /home/szw/lhm2/runtime/logs/kisaki-queue-nohup.log 2>&1 &

队列不停止或抢占任何已有进程。它只在 /home/szw/lhm2 内写入锁、日志、adapter 和结果。

## 可形成结论的条件

1. 两组 run manifest 显示相同 train/validation 哈希和固定验证集。
2. seed 42 自动质量门通过。
3. seeds 42/43/44 都完成并报告均值、标准差和失败记录。
4. Gold v2 人工审核、语义审计和冻结完成。
5. 匿名 A/B 人工盲评完成；AI 初审只能作为辅助意见。
