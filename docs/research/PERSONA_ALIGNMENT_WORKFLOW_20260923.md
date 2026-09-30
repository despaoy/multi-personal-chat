# 人物偏好数据、生成盲评与奖励裁判工作流

这轮在现有 DPO 训练链路之外补齐三项工具：真实候选采样、生成结果的盲评闭环、绑定人物证据的外部模型裁判。入口为 `backend/training/persona_alignment.py`，在项目根目录设置 `PYTHONPATH=backend` 后运行。

没有调用实际角色模型或付费裁判，没有制作/批准正式人物偏好数据，没有启动 8B 训练。集成测试使用合成场景与本机 HTTP 测试服务；它验证接口，不证明人物效果。后续已补齐 PPO rollout、critic、策略更新、保存恢复与盲评接入，运行说明见 [PPO 操作手册](PERSONA_PPO_RUNBOOK_20260923.md)。

## 流程

```text
已审核 train 场景 → 当前模型真实采样多个候选 → 匿名 A/B 审核
                                              ↓ 人工完成并锁定
                                   导出带内容指纹的偏好对 → 来源分组冻结 → DPO

独立 validation/test 场景 → SFT 与 DPO 各生成一条 → 匿名 A/B 审核
                                                     ↓ 人工完成并锁定
                                           胜/负/平、分维度与严重错误报告

同一批 validation 候选 → 外部人物裁判 → 与锁定人工决定比对 → 奖励校准报告
```

校准只允许 validation。test 可用于最终生成盲评，不能反过来导出训练数据或调校奖励。分组依据包含场景、来源 ID、证据内容；近义改写仍需额外人工检查。

## 1. 场景输入

输入 JSONL，每行一个场景。以下是**待审核模板**，`review_status=pending` 会被采样器拒绝；只有实际审核后才能改成 approved。

```json
{
  "id": "scene-001",
  "persona": "kisaki",
  "source_group": "original-scene-001",
  "source_ids": ["original-source-001"],
  "split": "train",
  "review_status": "pending",
  "persona_profile": "填写经过核验的人物画像",
  "relationship": "填写系统确认的对话者关系，不采用用户自称作为事实",
  "situation": "填写当前场景、人物目标与剧情进度",
  "evidence": [
    {"id": "e1", "source_id": "original-source-001", "content": "填写真实原文证据"}
  ],
  "history": [
    {"role": "user", "content": "填写此前用户消息"},
    {"role": "assistant", "content": "填写对应人物回复"}
  ],
  "user_message": "填写当前用户消息"
}
```

`history` 必须是完整的 user/assistant 轮次；无历史用空列表。未知事实场景允许 `evidence=[]`。画像、关系与情境是已审核的应用状态；原文证据通过项目现有 `build_grounded_user_message()` 放入转义的不可信参考区。所有模型使用完全相同的上下文快照。

这是固定历史下的下一条回复实验，不是自动滚动多轮对话，也不会在线检索 RAG 或自动取生产记忆。要测生产路径，应先保存该路径实际给模型的关系、证据和历史，再核对实验输入一致性。场景身份与 source_ids 应保持真实来源，不能通过改 ID 把 Gold 测试题变成训练题。

## 2. 真实采样

配置 JSON 示例；`revision` 记录实际基座/adapter 版本或权重哈希，工具不会替远端服务核验权重。model 名称需与已部署服务一致。

```json
{
  "models": [
    {
      "name": "sft",
      "model": "kisaki-sft",
      "revision": "填写实际模型与adapter版本",
      "base_url": "http://127.0.0.1:8000/v1",
      "api_key_env": "PERSONA_MODEL_API_KEY"
    }
  ],
  "generation": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256, "seed": 42},
  "samples_per_model": 4
}
```

无鉴权的本机服务省略 `api_key_env`。密钥值只读指定环境变量；禁止把 `api_key` 或含凭据的 URL 写进配置。服务应支持标准聊天补全参数；seed 会传入并记录，但不保证服务端实际执行确定性采样。

```bash
python -m training.persona_alignment sample \
  --scenes /path/to/reviewed-train-scenes.jsonl \
  --config /path/to/sampling.json \
  --output-dir /path/to/new-candidate-run
```

每个候选保留请求、模型版本、场景、输入、回复及各自哈希。不生成 chosen/rejected 标签；不因失败补固定模板。API 返回截断回复、空内容或异常会记为失败。候选逐条落盘，崩溃后的不完整日志不能直接用于评测；当前不自动续跑或重试，请排查后在新目录执行完整实验。

历史 `data/gen_preference_pairs.py` 的真实训练分支已禁用，防止继续从 Gold 取题并用模板负例训练。其 `--mock` 输出明确标记 mock，训练入口拒绝此类数据。

## 3. 候选盲评、锁定与导出

```bash
python -m training.persona_alignment blind \
  --run /path/to/new-candidate-run --purpose preferences \
  --output-dir /path/to/new-review
```

产生 `BLIND_REVIEW.md`、`blind_review.json`、`decisions.template.json` 和单独的 `blind_key.json`。只把前三个文件交给审核者，锁定决定前不打开 key。文件分离是程序化盲评约定，不是加密访问控制。

每个比较需要填写：七维比较、A/B 严重错误列表、总体 A/B/tie/invalid、理由、审核者和审核方式。无严重错误也必须主动填 `[]`。`human_confirmed` 默认 false，只有实际人工确认后才改为 true；AI 辅助审核使用 `human_confirmed_ai_assisted`，不得声明独立人工审核。

```bash
python -m training.persona_alignment lock \
  --packet /path/to/new-review/blind_review.json \
  --decisions /path/to/completed-decisions.json \
  --output /path/to/decisions-locked.json

python -m training.persona_alignment export \
  --packet /path/to/new-review/blind_review.json \
  --locked /path/to/decisions-locked.json \
  --key /path/to/new-review/blind_key.json \
  --output-dir /path/to/new-preferences
```

`lock` 不读取 key。导出拒绝错配的上下文、回复、决定或身份映射；平局、无效比较和优选回复仍有严重错误的记录不进入训练。与现有训练契约一致，同一 prompt 最多导出一对，其余比较记入排除统计；这一版不支持人工改写回复后直接沿用旧审批，改写需重新生成绑定并复审。

导出的 `reviewed_preferences.jsonl` 使用 `persona-preference-v1` 对话格式，可进入上一轮的 DPO loader 和 `scripts/prepare_kisaki_dpo_v3.py`。冻结器按来源成组切分；更改 split 不改变内容绑定，但更改回复或来源会失败。

## 4. SFT 与 DPO 生成效果比较

使用独立 validation 场景，采样配置包含 `sft` 与 `dpo` 两个模型，`samples_per_model=1`。保持所有解码参数一致。

```bash
python -m training.persona_alignment blind \
  --run /path/to/sft-dpo-eval-run --purpose evaluation \
  --baseline sft --candidate dpo --output-dir /path/to/new-eval-review
```

按上一节完成人工锁定，再运行：

```bash
python -m training.persona_alignment report \
  --packet /path/to/new-eval-review/blind_review.json \
  --locked /path/to/eval-decisions-locked.json \
  --key /path/to/new-eval-review/blind_key.json \
  --output-dir /path/to/new-eval-report
```

报告包括总体胜/负/平/无效、七维比较、严重错误计数、逐场景新增严重错误以及按相连来源组重采样的置信区间。平局计半分，但同时保留原始计数；只有一个独立来源组时不计算区间。小样本区间不稳定，报告不会自动批准模型发布。

## 5. 裁判与人工偏好校准

裁判配置是采样配置中的单个 model 对象，建议固定版本；真实调用由执行该命令的人选择服务与密钥。裁判只看到场景与回复，不看到生成模型身份。其 JSON 模式必须受到服务支持。

```bash
python -m training.persona_alignment judge \
  --run /path/to/sft-dpo-eval-run \
  --model-config /path/to/judge-model.json \
  --output /path/to/new-judgments.jsonl

python -m training.persona_alignment calibrate \
  --packet /path/to/new-eval-review/blind_review.json \
  --locked /path/to/eval-decisions-locked.json \
  --key /path/to/new-eval-review/blind_key.json \
  --judgments /path/to/new-judgments.jsonl \
  --output-dir /path/to/new-calibration
```

裁判输出七维分数、三个硬错误检查、逐维理由和真实证据 ID。信息不足可弃权；格式错误、越界数值或引用不存在的证据直接失败。所有分数均绑定具体回复与上下文，维持 `feedback_source=ai`、`review_status=pending`，不能自动变成人工标签。

校准报告同时给出覆盖率和被弃权/无效的数量，避免只展示容易样本上的一致率。它衡量裁判与人工是否一致，不代表奖励已对长度偏差、口癖投机或新场景泛化验证充分。

## 验证入口

```bash
python -m pytest backend/tests/test_persona_alignment_workflow.py backend/tests/test_persona_alignment_http_cli.py
```

旧 DPO/ORPO 的真实 CPU 集成测试也应与本轮回归共同运行。正式 GPU 训练、真人审核和实际裁判校准仍需真实数据、可用模型服务与审核结果。

2026-09-23 本轮验证：87 项相关测试通过，包含上述专项测试、偏好数据/奖励校验和微型 Qwen3 的真实 CPU DPO/ORPO 集成。静态检查与 CLI 帮助入口通过。未执行全仓库测试或 8B CUDA 验证。
