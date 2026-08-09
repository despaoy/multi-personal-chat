# 月社妃角色数据

当前唯一活动研究体系为 KISAKI-LLM-RESEARCH-V4。人物画像已确认，system prompt v3、训练候选、验证候选和 Gold 仍需按审核包逐项批准。

## 活动资产

| 内容 | 路径 | 用途 |
|---|---|---|
| 原始可追溯台词 | `tsukiyashiro_kisaki_raw.jsonl` | 来源覆盖和证据定位 |
| 原作训练候选 | `tsukiyashiro_kisaki_sft.json` | 人工审核输入 |
| 构造训练候选 | `experiments/train_v5_clean.jsonl` | 人工审核输入 |
| V4 审核包 | `../../../docs/research/review_packets/kisaki_v4/` | 用户逐批审核 |
| 人物提示词 | `kisaki_system_prompt_v3.txt` | 角色身份、关系、性格和表达 |
| V4 正式数据 | `experiments/v4/` | 审核并冻结后才会生成 |
| Gold v2 | `../../evaluation/kisaki_gold_set_v2.json` | 开发评测，不回流训练 |
| Gold v3 | `../../evaluation/kisaki_gold_set_v3.json` | 最终盲测，尚未生成 |

## 三层提示词

1. 人物层：`kisaki_system_prompt_v3.txt`。
2. 全局事实与安全层：`backend/inference/prompt_policy.py`。
3. RAG 证据层：仅检索命中时由同一策略模块条件注入。

人物提示词不承载密钥保护、管理权限或引用格式；RAG 内容被标记为不可信证据，不能覆盖系统规则。

## 审核与训练门禁

```bash
python scripts/build_kisaki_v4_review_packets.py --help
python scripts/validate_kisaki_v4_training_gate.py
```

训练数据、固定验证集、Gold v3 和 prompt v3 未全部批准并冻结前，不得启动正式 R1V4 训练。

## 历史资产

旧 E1/E2/E2'/E2'' 数据和补充集位于 `archive/legacy_e2/`，旧提示词、配置和结果位于实验目录的 `archive/`。这些资产只用于研究追溯，不被活动代码读取，也不用于当前结论。
