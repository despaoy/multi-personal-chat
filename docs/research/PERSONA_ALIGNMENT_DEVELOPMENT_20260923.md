# 人物偏好对齐开发：DPO 与 PPO 奖励准备

后续开发已补齐真实候选采样、生成盲评与外部人物裁判接入，见 [人物对齐完整工作流](PERSONA_ALIGNMENT_WORKFLOW_20260923.md)。本文保留第一轮 DPO 开发记录；PPO 策略训练循环仍未实现。

本轮实现 DPO 的可验证训练链路，以及供后续 PPO 使用的外部人物评分聚合与校准工具。没有启动月社妃正式训练，没有修改冻结训练数据或批准新的偏好标签；尚未实现 PPO rollout/critic/策略优化训练循环。

## 已实现

- 指定 SFT adapter 时，DPO 显式加载可训练 `default` 与冻结 `reference` 两份 adapter。检查初始权重完全一致、reference 不可训练、训练和验证后 reference 指纹不变。最终仅导出 policy adapter。
- 不存在、不完整或不符合 LoRA/bias 契约的 adapter 路径直接失败，不回退到基座。阻止覆盖已有 adapter 输出目录。
- 新增 `--eval-data`，运行训练前和训练后独立验证。报告原始评测值、环境依赖版本、数据数量、参考策略指纹。`eval_accuracy` 表示 TRL 的隐式奖励偏好准确率，**不是生成人物胜率**；SFT policy 与 reference 初始相同，DPO 初始奖励差为零，初始准确率可能为零，不能解释成 SFT 人物质量为零。
- 校验 train/validation 的 prompt、ID、source_group、scene_id、source_ids 和证据内容指纹隔离。最终 `test` 分区不能送入训练或训练期验证。文本近义重复仍需审核，当前不会把语义去重能力夸大为已实现。
- R4 冻结器按相互连接的来源/场景/证据分组切分，保持输入顺序无关的确定性；缺少分组元数据或仅有一个独立组时直接失败。验证比例近似 20%，完整组不拆开。冻结目录升为 `frozen_preference_v2`，旧目录保留。
- 对普通文本与对话格式检查 prompt 和完整回复长度，拒绝静默截断。R4 启动脚本默认预算 4096/3072，可由 `DPO_MAX_LENGTH`、`DPO_MAX_PROMPT_LENGTH` 覆盖，需按显存和实际数据长度选择。
- PPO 准备：七维人物奖励聚合，事实编造、关系越界和状态矛盾作硬失败；奖励不从口癖或关键词推断。提供独立人工偏好校准 CLI，输出一致率、平局与逐对结果，不自动宣布奖励可用于 PPO。

## 环境

使用独立训练环境，先安装适合机器的 PyTorch，再安装 `backend/requirements-preference.txt`。该文件固定 TRL 0.23.1、PEFT 0.17.1、Transformers 4.56.2、Accelerate 1.10.1、Datasets 4.1.1。4-bit CUDA 训练另需兼容 bitsandbytes。

不建议直接用这些依赖覆盖正在运行的业务/RAG 环境。其他 TRL 版本只有满足显式参考 adapter 接口时才允许该 SFT→DPO 路径，不能依赖其默认 reference 行为。

## DPO 使用

在项目根目录设置 `PYTHONPATH=backend` 后：

```bash
python -m training.preference_trainer \
  --data /path/to/train.jsonl \
  --eval-data /path/to/validation.jsonl \
  --base-model /path/to/Qwen3-8B \
  --adapter /path/to/approved-sft \
  --method dpo --epochs 1 --learning-rate 5e-7 --beta 0.1 \
  --max-length 4096 --max-prompt-length 3072 \
  --output-dir /path/to/new-dpo-run
```

CPU/BF16 流程验证可加 `--no-4bit`，其余精度与显存仍取决于模型及硬件。`--mock` 只验证流程，不报告人物质量。未传 `--eval-data` 的旧调用仍可运行，但报告清楚注明未计算独立偏好准确率。

普通文本偏好记录示意（须实际审核，不能把示例当训练数据）：

```json
{
  "id": "pair-001",
  "prompt": "已包含人物、关系和场景的完整输入",
  "chosen": "经过审核的优选回复",
  "rejected": "当前模型的真实问题回复",
  "review_status": "approved",
  "annotator": "manual",
  "metadata": {
    "persona": "kisaki",
    "source_group": "original-scene-001",
    "source_ids": ["original-evidence-001"],
    "split": "train"
  }
}
```

验证集同结构、不同来源，`metadata.split=validation`。已有 `contextual-evidence-v1` 对话导出也支持 validation 分区。历史未标明 split 的普通文本格式仍允许加载，但显式标成 test 的记录会失败。

## PPO 奖励工具

外部人工或模型裁判须提供完整七维分数，每项 [0,1]：

`persona_decision / relationship / emotion / grounding / continuity / naturalness / responsiveness`

同时明确三个布尔检查：

`fabricated_fact / relationship_violation / state_contradiction`

`training.persona_reward.aggregate_reward()` 接收 `judge_id`、`scores`、`violations`，输出 [-1,1] 标量和逐项贡献。任一硬失败为 -1。默认等权仅是可解释基线，使用自定义权重时必须包含全部维度。

校准 JSONL 每行：`id`、`source_group`、`split="validation"`、`human_approved=true`、`human_preference="a"|"b"|"tie"`、`a` 与 `b` 两份上述评分对象。每一对必须针对相同上下文，并在源标注资产中保留实际候选文本、证据和裁判版本；本工具不负责自动生成或人工批准这些判断。

```bash
python -m training.persona_reward \
  --calibration-data /path/to/human-reviewed-calibration.jsonl \
  --output /path/to/new-calibration-report.json
```

可加 `--weights /path/to/weights.json`。报告不覆盖已有文件；输出 `measured_not_approved_for_ppo`，防止把一批样例的一致率自动转换成上线或训练许可。之后仍需检查长答偏好、口癖投机、过度亲密、无条件拒答及新场景泛化。

## 验证

纯逻辑回归：

```bash
python -m pytest backend/tests/test_preference_alignment_contracts.py backend/tests/test_persona_reward.py backend/tests/test_preference_compat.py backend/tests/test_evidence_training_dataset.py
```

安装独立训练依赖后执行真实 CPU 集成：

```bash
python -m pytest backend/tests/test_preference_real_cpu.py backend/tests/test_prequantized_training.py
```

真实集成测试在临时目录随机创建微型 Qwen3 和非零 SFT LoRA，运行两步 DPO、独立验证、冻结参考检查、导出与重载；不下载模型，不读取正式角色语料。它验证工程链路，不证明 8B/4-bit CUDA 兼容性或人物效果。

本轮验证记录（2026-09-23）：独立训练环境的相关回归与真实 CPU 集成测试通过；覆盖普通文本 DPO、对话格式 DPO、ORPO、量化准备顺序、来源分组冻结、参考完整性、保存重载和奖励校准。报告还记录训练/验证记录的 SHA-256，便于追溯实际输入。

## 下一阶段

在合格 SFT 和已审核偏好数据就绪后运行 DPO pilot，并使用独立生成盲评确认人物、关系和事实表现。后续开发已实现绑定候选与证据的裁判、多轮 rollout、critic、PPO 策略更新和检查点恢复；完整使用说明见 [PPO 操作手册](PERSONA_PPO_RUNBOOK_20260923.md)。奖励聚合工具本身仍只是训练器的一部分，不能替代真实校准和生成盲评。
