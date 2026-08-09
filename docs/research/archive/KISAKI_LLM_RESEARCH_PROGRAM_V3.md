# 月社妃 LLM 研究主线 v3（历史归档）

> 历史设计文档。当前执行入口已经迁移到 [V4 人工审核与重训练](KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md)。本文中的 826/92 数据契约和旧启动命令不得用于新的正式结论。

## 1. 研究问题

项目不再把功能数量作为成果，而围绕一条可验证证据链展开：

`数据与评测可信性 -> PEFT 单变量消融 -> RAG 分层实验 -> 推理性能 -> DPO pilot -> 多平台系统闭环`

月社妃是唯一正式研究角色。Minamo 与胡桃只用于各 20 条部署、路由和外部有效性演示，不据此声称跨角色泛化。

## 2. 统一编号与当前状态

| 编号 | 内容 | 当前状态 | 正式结论 |
|---|---|---|---|
| R0 | 旧 canonical 数据、人物画像、Gold v2、prompt v2 | 已完成，现作为 V4 开发与历史材料 | 旧数据含已知质量问题 |
| R1-E1 | 标准 LoRA | Seed 42 已训练、评测 | 历史对照 |
| R1-E2 | 仅 NEFTune alpha=5 | Seed 42 已训练、评测 | 历史对照 |
| R1-E3 | 仅 DoRA | Seed 42 已训练、评测，旧自动指标最佳 | 历史对照 |
| R1-E4 | 仅 RSLoRA | Seed 42 已训练、评测，质量门失败 | 历史负结果 |
| R1-E5 | 仅 Sequence Packing | Seed 42 已训练、评测，安全诊断失败 | 历史负结果 |
| R2 | 四种检索与三种回答策略 | 60 条候选集已生成，待人工冻结 | 无 |
| R3 | BF16、AWQ、动态 LoRA、Merge | 流式基准器已实现，待隔离运行 | 无 |
| R4 | QLoRA DPO pilot | 被 100 条人工批准门禁阻塞 | 无 |
| S1 | 路由、traceId、AstrBot | 工程链路已有，正式系统实验待执行 | 无 |

旧 E1/E2/E2'/E2'' 统一为 `legacy_exploratory_non_comparable`。它们保留用于失败分析，但不能出现在 v3 正式消融表中。

## 3. 历史实验契约

旧 R1 五组共同固定：Qwen3-8B-Instruct BF16、826 条训练、92 条验证、Seed 42、LoRA r=32/alpha=64、七个 target modules、3 epochs，以及相同学习率、批次预算和生成参数。该数据含 111 条已确认存在风格偏移的 LLM 样本，训练器还只监督了多轮样本最后一个 assistant 回复，因此不得作为 V4 正式训练契约。

每个正式结果必须满足：

- `mock=false`；
- 记录 Git commit、数据、prompt、模型、adapter 与生成参数哈希；
- 记录 GPU、CUDA、Python 和依赖版本；
- 保留每条样本输出和错误；
- 负结果不得删除；
- 人工盲评前隐藏模型身份，AI 意见只能作为辅助。

Seed 42 的 prompt-v1 盲评已作为不可修改 pilot 保存。新结论只使用 prompt v2；E1/E2 不重训，只重新生成。

## 4. 旧 R1 执行方式（禁止用于 V4）

以下命令只保留用于复现实验历史，不应再次执行。V4 必须先通过人工审核门禁。

```bash
cd /home/szw/lhm2/qqchat-enhanced
/home/szw/lhm2/envs/qqchat-gpu-qwen3/bin/python scripts/validate_kisaki_experiments.py \
  --require-model --formal-eval
python scripts/validate_kisaki_v4_training_gate.py
```

门禁当前应返回 blocked，这是预期行为。审核未完成、V4 数据未冻结或 Gold v3 不存在时，不得启动 GPU 训练。

评测完成后，为 E1 与 E2/E3/E4/E5 各生成 40 条分层盲评。先完成四个小盲评，再选最优候选与 E1 做 120 条完整盲评。质量门失败是有效负结果，不等于运行失败。

## 5. R2 执行门槛

`kisaki_rag_eval_v2_candidates.json` 当前是候选集：30 条单证据、15 条多证据、15 条无答案。必须人工检查问题、答案、证据 ID 与无答案标签，改为 `status=frozen`、`formal_use_allowed=true` 并生成新哈希后，正式命令才会运行。
审核完成后运行：

```bash
python scripts/freeze_kisaki_rag_v2.py --input backend/data/character_dialogues/experiments/research/kisaki_rag_eval_v2_candidates.json --output backend/data/character_dialogues/experiments/research/kisaki_rag_eval_v2.json
```

检索层先比较 vector、BM25、hybrid、hybrid+reranker，并记录 Recall@1/5、MRR、nDCG@5 和延迟。选定检索器后，再比较直接回答、置信度拒答、Corrective RAG；RAG 数据永不加入 LoRA 训练集，引用只通过结构化字段展示。

## 6. R3 计量规则

每个模型使用独立 vLLM 进程，固定 60 条 prompt，预热 5 次、正式 3 轮，并发为 1/4/8。TTFT 来自 SSE 首个内容 token，不能用整次请求耗时代替。原始测量必须保留，并同时报告 inter-token latency、tokens/s、P50/P95/P99、启动时间、峰值显存、失败率和质量差。

AWQ + 动态 LoRA 只有在兼容性检查通过时才进入正式表；否则如实记为 unsupported，而不是补造数值。

## 7. R4 与 S1 门禁

R4 只做一次 DPO pilot，不并行做 ORPO。输入必须至少包含 100 条月社妃人工最终批准偏好对，随后固定拆成 80% 训练与 20% held-out。准备命令：

```bash
python scripts/prepare_kisaki_dpo_v3.py \
  --input /home/szw/lhm2/runtime/datasets/kisaki_preferences_reviewed.jsonl \
  --output-dir /home/szw/lhm2/runtime/experiments/kisaki/r4/seed42/data
```

条件不足时命令返回 blocked，不产生训练数据。正式 pilot 使用 `scripts/lab-run-kisaki-r4-dpo.sh GPU BEST_R1_ADAPTER PREFERENCE_SOURCE`：脚本会复核冻结清单与训练集哈希、以可训练方式加载最佳 SFT adapter，并固定 `learning_rate=5e-7`。held-out 文件不会传给训练器，训练后必须单独完成 chosen/rejected 评分与盲评。QA-LoRA 保持 `not_implemented`，不得以 QLoRA 冒充。

S1 最终检查手动选择、规则路由、意图路由、adapter 兼容性、fallback 和 traceId。AstrBot 至少覆盖 QQ 私聊、群聊、重复消息、超时和降级；个人微信仅用测试账号并披露稳定性与合规限制。

## 8. 展示时的结论边界

只有依次达到“训练成功、自动评测完成、Gold 冻结、盲评完成、可形成结论”后，结果才能进入答辩结论表。单种子结果称为 pilot；多种子补充只对 E1 和最终最佳候选执行。实验没有提升同样是结果，重点解释技术、成本、稳定性和误差类型之间的权衡。
