# 初学者真实 LLM 实验操作指南

> 当前基线：官方 `Qwen/Qwen3-8B`、Python 3.12、PyTorch 2.8、vLLM 0.10.2、RTX 3090。本地目录为兼容旧部署可继续命名 `Qwen3-8B-Instruct`。
> Qwen2.5 的既有结果属于历史对照，见 `archive/REAL_VLLM_BENCHMARK_REPORT.md`。

## 1. 这套实验在证明什么

你不是只演示“模型能回答”，而是要回答四个研究问题：

1. 数据治理是否提高角色一致性并减少安全退化？
2. LoRA、DoRA、RSLoRA、NEFTune 的质量、显存和训练成本如何权衡？
3. AWQ 与非量化推理在 TTFT、吞吐、P95 和显存上有什么差异？
4. vector、BM25、hybrid、reranker 哪种 RAG 方案更可靠，什么时候应拒答？

## 2. 开始前检查

登录服务器后执行：

```bash
source "$MULTIPERSONAL_LAB_ROOT"/activate_qqchat.sh
cd "$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"

python --version
python -m pip check
nvidia-smi
git status -sb
```

你要确认：

- Python 为 3.12。
- `pip check` 无依赖冲突。
- 目标 GPU 至少有约 20GB 可用显存。
- Git 工作区的重要数据已经提交或归档。
- 模型存在于 `"$MULTIPERSONAL_LAB_ROOT"/runtime/models/`。

## 3. 理解四类目录

| 内容 | 目录 |
| --- | --- |
| 代码、配置模板、可提交数据 | `"$MULTIPERSONAL_LAB_ROOT/multi-personal-chat"/` |
| 基座、AWQ、Embedding、Reranker | `"$MULTIPERSONAL_LAB_ROOT"/runtime/models/` |
| LoRA checkpoint 和 final | `"$MULTIPERSONAL_LAB_ROOT"/runtime/loras/` |
| 日志、评测和临时输出 | `runtime/logs/`、`runtime/results/`、`runtime/tmp/` |

不要把 AWQ 模型用于普通 SFT。训练使用官方 `Qwen/Qwen3-8B` 权重，AWQ 版本 `Qwen/Qwen3-8B-AWQ` 主要用于推理基准。

## 4. 先验证数据隔离

训练前必须确认 Gold Set 没有进入训练集：

```bash
python scripts/validate_kisaki_v4_training_gate.py
python scripts/validate_kisaki_v4_training_gate.py
python -m pytest backend/tests/test_character_benchmark.py -q
```

至少保留：

- 固定 train/eval 拆分。
- Gold Set 独立文件和内容哈希。
- 合成数据的来源、模型、prompt 和审核状态。
- 重复样本、过长样本、安全样本和 RAG 引用样本统计。

## 5. 跑一个最小训练

先检查审核和冻结门禁，不要直接启动长实验：

```bash
python scripts/validate_kisaki_v4_training_gate.py
```

当前应看到阻塞项，这是为了防止未审核数据进入训练。全部审核并冻结后，使用：

```bash
python scripts/run_kisaki_experiment.py --experiment e1 --seed 42
```

正式训练应保存：

- 训练配置 JSON。
- TensorBoard 日志。
- train/eval loss。
- 峰值显存和训练时长。
- 最佳 checkpoint 与 `final/`。
- 数据哈希、代码提交和随机种子。

月社妃旧直启脚本已归档且禁止使用。先运行 `scripts/validate_kisaki_v4_training_gate.py`；门禁通过后只使用 `scripts/run_kisaki_experiment.py` 启动 R1V4。

## 6. 启动真实 vLLM

先确认端口和 GPU：

```bash
nvidia-smi
curl -fsS http://127.0.0.1:8001/v1/models || true
```

默认服务使用 8001，独立 LoRA 对比实验可使用 8002。不要同时让两个 vLLM 进程争用同一张 3090。

基础 AWQ 示例：

```bash
bash scripts/lab-start-vllm-daemon.sh
```

模型列表：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
```

## 7. 运行评测

每个候选模型必须使用同一 Gold Set、生成参数和随机种子。记录：

- 角色一致性和人工盲评胜率。
- 格式正确率、重复率、Distinct-1/2。
- 安全通过率、RAG 引用准确率和拒答质量。
- TTFT、平均延迟、P95/P99、吞吐和显存。

示例：

```bash
PYTHONPATH=backend python -m evaluation.character_benchmark_v3 --help
```

正式评测必须传入同一冻结 Gold、同一 prompt v3、`--compose-runtime-policy` 和同一生成参数。不要再使用旧 E2'' 模型名或旧结果路径。

命令参数以 `--help` 输出为准；不要为了匹配文档而忽略脚本真实接口。

## 8. 人工盲评

1. 隐藏模型名称并随机交换 A/B。
2. 每条只看 prompt、A、B 和评分规则。
3. 选择 A、B、平局或两者都失败。
4. 记录理由标签：角色、事实、安全、引用、重复、格式。
5. 解盲后统计总体和分类型胜率。

AI 初审只能帮助发现明显问题，不能替代最终人工结论。发送到第三方 API 前要移除隐私和密钥。

## 9. LoRA 消融矩阵

一次只改变一个主变量：

| 组别 | 变量 |
| --- | --- |
| Rank | r=8/16/32/64 |
| Alpha | 固定 rank 后比较 alpha/r 比例 |
| Modules | attention-only 与 all-linear |
| Method | LoRA、DoRA、RSLoRA |
| Regularization | NEFTune 开/关 |
| Data | 原始、清洗、困难负例、RAG 增强 |

DoRA 与 RSLoRA 是不同的参数化/缩放策略。技术上某些 PEFT 版本允许组合，但研究消融应先分别启用，再把组合列为独立实验，不能把组合结果归因给其中一个方法。

## 10. RAG 实验

固定 query 集，比较：

1. vector。
2. BM25。
3. hybrid。
4. hybrid + reranker。
5. Corrective RAG。

每组报告 Recall@K、MRR、nDCG、引用准确率、拒答准确率、检索延迟和失败样本。没有可靠证据时，拒答比编造更好。

## 11. 结果记录模板

每次实验建立唯一编号并保存：

```text
experiment_id
git_commit
dataset_hash
gold_set_hash
base_model
adapter_config
seed
hardware
command
start/end time
train/eval metrics
latency and memory
raw outputs
failure analysis
conclusion and limitation
```

## 12. 常见错误

- **CUDA OOM**：停止其他 GPU 任务，降低 batch、序列长度或显存利用率。
- **LoRA 扫描不到**：检查 `LORA_PATH`，并确认 adapter 目录含 `adapter_config.json` 和权重。
- **LoRA 不兼容**：Qwen2.5 adapter 不能直接挂载到 Qwen3。
- **vLLM 模型不存在**：请求中的 model 名必须与 `/v1/models` 一致。
- **训练成功但效果差**：先检查数据泄漏、label mask、截断和验证集，不要先盲目增大 rank。
- **指标很好但样本很差**：指标实现或阈值可能有缺陷，必须抽查原始输出。

## 13. 推荐执行顺序

1. 固定数据版本与 Gold Set。
2. 复现 LoRA baseline。
3. 完成 r/alpha/modules 消融。
4. 分别测试 DoRA、RSLoRA、NEFTune。
5. 做人工盲评和失败分类。
6. 做 RAG 检索对比。
7. 做 AWQ/FP16/动态 LoRA 推理基准。
8. 将最佳方案接入 FastAPI 和 AstrBot，录制 traceId 全链路演示。
