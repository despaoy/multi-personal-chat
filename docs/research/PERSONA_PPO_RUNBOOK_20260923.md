# 人物 LoRA / DPO / PPO 基本闭环

工程入口已经覆盖：审核场景、候选生成、盲评与锁定、偏好对导出、DPO、奖励校准、PPO 多轮采样与更新、检查点恢复，以及训练后生成盲评。没有自动批准正式数据，也没有完成真实人物效果实验。

## 为了更像人物，先改善什么

1. 修复 SFT 数据配比。已有研究记录 V4 模拟助手内容占监督字符约 82.26%，现有 Qwen3-8B adapter 未通过人物/事实门槛。应先完成 V5 原文与情境数据审核和重新训练，不能期待 PPO 自动恢复被通用助手话术覆盖的人物行为。
2. 把人物的判断、关系边界、当前情绪、已知事实和连续状态写进经过审核的场景。同一人物面对熟人、陌生人和冲突时应有可解释的差异；口癖只能作为辅助。
3. DPO 使用同场景真实生成并经人工比较的回复。优选依据是人物会怎样判断和行动，而不是回复更长或含有更多人物词汇。未知信息、不合理亲密关系、诱导出戏、跨轮承诺等场景都应覆盖。
4. PPO 在裁判与人类比较已校准后做小步实验。先和同一 SFT 起点的 DPO 分别比较，避免一次叠加多个变化而无法归因。当前 PPO 固定以传入的初始 adapter 为参考；若传入 DPO adapter，则参考也是该 DPO adapter，必须在实验记录中明确。
5. 用独立来源的生成盲评验证七维表现和严重错误。校准、调参只用 validation，最终 test/Gold 保持隔离。

论文依据与项目实践见 `PERSONA_LORA_DPO_PPO_RESEARCH_20260923.md`；审核、DPO 数据与盲评命令见 `PERSONA_ALIGNMENT_WORKFLOW_20260923.md`。

## 环境与输入

使用 `backend/requirements-preference.txt` 的固定依赖与适合硬件的 PyTorch。当前工程测试使用隔离环境 `.tmp/preference-validation`，CPU 上的小 Qwen3；8B、CUDA、bitsandbytes 和梯度检查点尚需目标机器实测。

在项目根目录设置 `PYTHONPATH=backend`。PowerShell 为 `$env:PYTHONPATH="backend"`；Linux 为 `export PYTHONPATH=backend`。

复制并填写：

- `backend/training/configs/persona_ppo.example.json`：基座、初始 adapter、全新输出目录。
- `backend/training/configs/persona_judge.example.json`：实际裁判服务与版本；密钥仅通过指定环境变量提供，无鉴权的本地服务可删除 `api_key_env`。
- 审核完成的 train/validation 场景 JSONL，格式与采样器一致。可附加 `"continuations": ["第二轮用户消息", "第三轮用户消息"]`；人物回复实际生成后加入后续历史。后续用户消息是预设脚本，未接入在线用户模拟器、生产 RAG 或记忆更新。
- `persona_alignment calibrate` 生成的 `report.json`，必须绑定同一个裁判模型、版本和评分规则。

本地基座权重、配置与 tokenizer 文件会读取并计算指纹；这会增加大模型启动时间。远端 Hugging Face 基座必须指定 `base_revision` 为完整 40 位 commit，加载时固定到该版本。

## 检查、训练、恢复

以下命令为单行示例，替换路径后执行。检查不会加载训练权重或调用裁判，但会读取模型文件与 tokenizer；远端 tokenizer 可能需要下载。

```bash
python -m training.persona_ppo --config /path/ppo.json --train-scenes /path/train.jsonl --validation-scenes /path/validation.jsonl --judge-config /path/judge.json --calibration-report /path/calibration/report.json --check-only
python -m training.persona_ppo --config /path/ppo.json --train-scenes /path/train.jsonl --validation-scenes /path/validation.jsonl --judge-config /path/judge.json --calibration-report /path/calibration/report.json
```

校准试验默认至少 30 对、10 个来源组、70% 一致率和 80% 覆盖率。这些是可配置的工程试验门槛，未经统计效能研究确认，不是奖励可靠或模型可发布的证明。只有明确开展未校准小试验时才设置 `allow_uncalibrated_pilot=true` 并省略校准文件；结果会保留该标记。

每次更新保存 `checkpoint-NNNNNN`。`--stop-after 1` 可在第一个完整检查点停下。恢复时，复制配置，仅更改输出目录和需要延长的 `updates`，保留其他字段、场景顺序及校准文件：

```bash
python -m training.persona_ppo --config /path/resume.json --train-scenes /path/train.jsonl --validation-scenes /path/validation.jsonl --judge-config /path/judge.json --calibration-report /path/calibration/report.json --resume /path/old-run/checkpoint-000001
```

恢复边界是完整策略更新，包含 actor、critic、优化器与随机状态；崩溃时尚未完成的更新会重新采样及调用裁判，远端非确定性仍可能改变结果。拒绝覆盖旧目录和检查点，检查点哈希用于误改检测，不是来源认证。

## 输出与评测

- `run.json`：配置、依赖版本、基座与数据绑定、奖励来源。
- `rollouts-NNNNNN.jsonl`：每轮真实上下文、回复、裁判依据与奖励；其中可能含原文和对话，应按训练数据管理。
- `checkpoint-NNNNNN/`：仅策略 adapter、价值头及可恢复训练状态；冻结参考从原始 adapter 加载并核验。
- `adapter/`：可独立加载的最终 LoRA adapter，不含参考 adapter 或 critic。
- `result.json`：训练状态、奖励/KL/裁剪等诊断。`persona_effectiveness=not_evaluated` 不会由训练奖励自动改为成功。
- `validation-NNNNNN/`：同一固定历史下的初始参考与当前 PPO 生成结果，兼容现有盲评工具。

```bash
python -m training.persona_alignment blind --run /path/run/validation-000010 --purpose evaluation --baseline sft_reference --candidate ppo_policy --output-dir /path/new-review
```

然后按工作流文档人工填表、lock、report。验证中的截断、空回复保留为失败；不把不完整实验当作胜率。比较 SFT/DPO/PPO 时可把三个独立服务放进现有采样配置，保持相同场景与解码参数，再分别创建基线对比。

## 实现边界

这是单设备、串行生成的 LoRA PPO 基线，不是分布式高吞吐训练器。策略采样固定 temperature=1、top_p=1、top_k=0，使行为概率与 PPO 目标一致；仅对人物生成 token（包括实际采样的 EOS）计算目标。裁判弃权会跳过整个多轮 episode；全批弃权则停止。达到长度限制或空回复使用明确的 -1 规则惩罚，不伪造裁判分数。价值头读取分离梯度的共享特征，价值损失不直接更新人物策略。

KL 惩罚与阈值是采样估计。策略 KL 在每轮优化前检查，参考 KL 在下一批采样后检查，不能保证单次更新不越界；异常或停止状态应查看最后检查点与盲评，不能自动部署。后续生成历史每轮重新检查上下文预算，超限报错，不静默截掉人物证据。

工程已验证不等于人物改善已验证。剩余实验工作是：V5 数据审核与 SFT 修复、正式偏好标签、真实裁判校准、目标 GPU 冒烟测试、SFT/DPO/PPO 对照训练、保留测试集盲评，以及生产对话路径验收。
