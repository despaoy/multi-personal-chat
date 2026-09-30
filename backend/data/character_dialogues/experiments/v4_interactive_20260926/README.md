# 审核后训练数据

train.jsonl 是去重替换后的完整普通训练集；sft.interactive.approved.jsonl 是本次原文增补；dpo.train.jsonl 是本次正式审核偏好对。

使用 sft_training_config.json 读取新训练集，验证仍使用原冻结验证集。未自动修改旧实验配置或启动任务。生产 DPO 仍需满足既有100对门槛。审核授权、替换记录与终审意见均保留在本目录。
