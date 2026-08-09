# 月社妃 V4 人工审核与重训练状态

## 当前状态

- 正式训练：`blocked_pending_human_review`
- 旧 R1 Seed 42：保留为历史对照，不覆盖、不删除
- 服务器 checkpoint：没有中间 checkpoint，保留 E1-E5 final adapter
- Gold v2：降级为开发评测集
- Gold v3：必须在训练数据冻结后生成

## 为什么需要 V4

旧正式训练集包含 111 条 `llm_v3_deepseek` 样本，存在元叙事过载、句式模板化和角色语气偏移。旧训练器对多轮数据只监督最后一个 assistant 回复，也会浪费早期回合。V4 先由项目负责人检查人物画像、原作提取、构造数据、验证集与 Gold，再重新进行单变量 PEFT 消融。

## 人工审核入口

从 [审核指南](review_packets/kisaki_v4/00_GUIDE.md) 开始，建议依次检查：

1. 人物画像和 system prompt。
2. 1,598 条原作台词与上下文。
3. 801 条原作训练候选。
4. 159 条构造训练候选。
5. 旧验证集和 V5 草案验证集。
6. Gold v2 开发集。
7. 排除样本和实验配置。

审核意见不会直接覆盖原数据。只有项目负责人明确确认的分类才会进入冻结数据集。

## 训练门禁

`scripts/validate_kisaki_v4_training_gate.py` 会检查：

- 必需审核分类是否全部明确通过。
- V4 train/validation 是否冻结并具有哈希。
- Gold v3 是否冻结且为 150 条。
- 可选的服务器剩余空间是否不低于 15GB。

门禁通过前，不生成正式 R1V4 配置，也不启动 GPU 训练。
