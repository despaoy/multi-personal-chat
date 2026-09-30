# 原文回复与 DeepSeek 回复构造 DPO 偏好

采用原文作为优选候选、普通模型输出作为劣选候选。模型接收相同的回复前上下文及人物信息，不接收目标台词或后续剧情。原文的出处真实，不意味着在信息不足的输入下也一定是可学习的正确答案。

## 当前产物

`backend/data/character_dialogues/experiments/source_dpo_draft_20260924/source_contexts.pending.jsonl` 包含 157 条待审核原文场景。

此次从冻结 V4 train 中筛选：排除 404 条非游戏提取记录、107 条需要单独处理的上下文结构、256 条与 V4 validation 场景/台词重叠的记录、2 条与 raw 原文无法精确对应的记录。不修改冻结数据，不自动修正原文。这里只检查传入的 V4 validation；正式使用时必须补充其他保留测试资产的来源重叠检查，不能据此宣称已检查全部 Gold。

每条保留原始训练记录、目标事件 ID、原文行事件指纹和精确台词。场景状态仍为 pending，不是已批准的训练数据。关系、情境、人物画像需按目标台词之前的剧情补齐；不得照抄默认“作品外用户”关系。对话者字段用于帮助核验真实原作身份。

## 执行流程

在项目根目录设置 `PYTHONPATH=backend`。准备入口：

```bash
python -m training.source_preferences prepare --train backend/data/character_dialogues/experiments/v4/train.jsonl --heldout backend/data/character_dialogues/experiments/v4/validation.jsonl --raw backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl --profile backend/training/configs/source_persona_profile.example.txt --output-dir /path/new-source-draft
```

审核待选场景的 `scene`，填写准确关系、情境、必要历史与证据；无法补齐或存在剧透的记录暂不选用。确认后填写 `scene.review_status=approved`、`context_review.reviewer` 及三个明确检查项。原始 `source_row`、`source_reply` 和指纹不可改写。将选中的记录保存为新的 reviewed JSONL。

配置 `backend/training/configs/source_preferences.deepseek.json` 使用 `https://api.deepseek.com/v1` 和 `deepseek-chat`。密钥只从 `DEEPSEEK_API_KEY` 环境变量读取，不进入配置或产物。`deepseek-chat` 是服务别名，配置明确记录 unpinned，不保证服务商长期保持相同权重，seed 也不保证远端确定性。

```bash
python -m training.source_preferences sample --reviewed /path/source-contexts-reviewed.jsonl --raw backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl --config backend/training/configs/source_preferences.deepseek.json --output-dir /path/new-source-run
python -m training.persona_alignment blind --run /path/new-source-run --purpose preferences --output-dir /path/new-source-review
```

每个场景只调用一次普通模型。原文记录显式标注 `source_excerpt`，不伪称模型生成；两条回复进入匿名审核。使用已有 lock/export 命令导出，只有原文胜出且无硬错误的比较进入训练。平局、原文信息不足、普通模型更合适的比较均不按该构造策略导出。不会把“不是原文措辞”单独当作劣选理由。

原文候选不能用作 SFT/DPO 模型生成效果评测，否则会把作者写作质量误当作模型表现。训练效果仍需另一批独立场景上真实生成的 SFT/DPO 对照。

## 初始准备阶段记录

本次已完成代码和真实原文待审提取，未向 DeepSeek 发起请求，也未自动确认场景或生成正式 chosen/rejected。29 项相关测试通过，包含原文核验、保留集排除、目标泄漏阻断、一次候选调用、盲评导出与反向偏好排除；外部服务的真实可用性尚未验证。

## 后续实际执行：原文检查与 DeepSeek 生成

用户随后明确要求由助手检查原文并生成数据。本轮调用 DeepSeek 实际完成 105 条回复生成；新增核对 V4 validation 行范围及 Gold V3 引用事件，157 条草稿中再排除 52 条。输入仅包含目标回复之前的原文、对话者和知识边界说明，不包含目标回复或后文。完整章节文件哈希、目标事件和原文行号均记录在产物中。

产物目录：`backend/data/character_dialogues/experiments/source_dpo_deepseek_20260924/`。

- `results.jsonl`：逐次调用结果和审核记录，只追加；重试按 ID 取最后一次结果，不能把 109 次记录当作 109 个独立样本。
- `FINAL_SUMMARY.json`：105 个场景完成生成，104 个获得有效裁判结果，1 个重试后仍失败。
- `preferences.ai-reviewed.pending.jsonl`：DeepSeek 按自报置信度 0.8 初筛的 2 对。这些分数没有经过校准。
- `second_review_bound.json`：助手对 11 对做第二次基于具体原文的复核，绑定记录哈希；保留 9 对，排除 1 对，因疑似原文缺字暂缓 1 对。不是独立人工盲评，也没有统计校准。
- `dpo.adjudicated.ai.jsonl`：最终建议的 9 对小规模 DPO 候选，覆盖 8 个来源组。
- `sft.adjudicated.ai.jsonl`：同样输入配原文输出的 9 条 SFT 候选。
- `ADJUDICATED_REVIEW.md`：最终复核理由和原文/生成回复对照；`REVIEW.md` 仅记录第一阶段机器筛选。

第二次复核没有机械沿用置信度门槛：部分低置信度样本有明确的新增设定、知识越界或关系错误，按证据保留；初筛中一条把“那就好”判作无承接的理由不成立，已排除。保留 9 对并不意味着剩余候选都错误或无价值，其他原文胜出但差异较弱的比较仍保留在原始日志。

所有导出明确标记 `feedback_source=ai`、`human_final_approved=false`、`review_status=pending`，不伪装成人工批准数据，不自动绕过现有正式训练入口，也未启动训练。数据量不足以评价效果或单独构造可靠验证集。生成与初筛使用同一 DeepSeek 服务，有自评偏差；助手的二次复核也仍是 AI 意见。当前验证覆盖上述 V4/Gold V3 资产的明确来源重叠，不宣称语义近似泄漏已经全部排除。

代码检查通过；相关 55 项测试通过，并额外核对 105 条记录内容指纹、9 对原文/回复和输入绑定。
