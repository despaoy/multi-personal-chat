# R1V4 配置审核

## 数据判断

当前 canonical train 保持 926 条记录、1,961 个有效监督目标，不增加或删除数据：

- 522 条原作记录负责语言、关系和原作行为锚定；
- 150 条已审核构造记录补充日常与关系情景；
- 254 条已审核五轮会话补充通用任务和短期多轮一致性；
- 70 条固定 validation 全部来自原作，只用于训练期 loss，不单独代表五维角色质量。

数据数量足以支持单角色 LoRA 基线。首轮 E1 的标签覆盖、截断和数据绑定均正常，但生成出现过短回答和语义丢失，因此本轮不通过扩充数据解决，而是先降低更新强度。

## 稳定性修订 v2

首轮 E1 pilot 使用 `learning_rate=2e-4`、3 epochs，在约 1.73 epoch 后 validation loss 开始反弹，且正式生成门禁检测到回复塌缩。统一基础配置调整为：

| 参数 | 首轮 pilot | 稳定性修订 v2 |
|---|---:|---:|
| learning rate | 2e-4 | 1e-4 |
| epochs | 3 | 2 |
| checkpoint 保留数 | 1 | 4 |

`r=32`、`alpha=64`、7 个 target modules、BF16、有效 batch 8、Seed 42、数据和 Prompt 均保持不变。保留多个 checkpoint 是为了同时比较 validation loss 和开发集自由生成质量，不能再只根据最低 loss 判断最佳模型。

## 单变量矩阵

| 实验 | NEFTune | DoRA | RSLoRA | Packing | checkpoint 保留数 |
|---|---:|---|---|---|---:|
| E1 | 0.0 | False | False | False | 4 |
| E2 | 5.0 | False | False | False | 4 |
| E3 | 0.0 | True | False | False | 4 |
| E4 | 0.0 | False | True | False | 4 |
| E5 | 0.0 | False | False | True | 4 |
