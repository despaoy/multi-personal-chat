# R1V4 配置审核

> 新 V4 配置要等数据冻结后生成。以下展示旧 V3 差异和已经确定的 checkpoint 策略。

| 实验 | NEFTune | DoRA | RSLoRA | Packing | 旧保留数 | V4 保留数 |
|---|---:|---|---|---|---:|---:|
| E1 | 0.0 | False | False | False | 3 | 1 |
| E2 | 5.0 | False | False | False | 3 | 1 |
| E3 | 0.0 | True | False | False | 3 | 1 |
| E4 | 0.0 | False | True | False | 3 | 1 |
| E5 | 0.0 | False | False | True | 3 | 1 |
