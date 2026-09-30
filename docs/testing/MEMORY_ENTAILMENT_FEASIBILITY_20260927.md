# 专用NLI是否能补足人物归属：离线可行性结果

2026-09-27，r93。结论：本次候选不适合作为自动事实准入/撤回依据，未接入运行时。

## 为什么试验

r90–92表明完整原话、小上下文、标准聊天模板和未量化基座都没有消除人物归属错误。写入审计另表明：规则提取器对隐含否认没有事实候选，规则模式就不写版本变更；SQLite现有显式RETRACT目标操作本身能撤回，但不能凭“那个”猜目标。此处测试专门训练的自然语言推断模型，而不是增加同一8B模型的提示词/复核次数。

读取[模型卡](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)：mDeBERTa-v3-base，以MNLI与XNLI专业翻译开发集训练；报告中文XNLI准确率0.8116。此为作者自己的基准，不是本项目人物对话成绩。模型卡链接[DeBERTa-v3](https://arxiv.org/abs/2111.09543)与[XNLI](https://arxiv.org/abs/1809.05053)，本轮没有声称完整阅读/复现两篇论文。NLI的entailment/neutral/contradiction区分文本支持、证据不足和矛盾，不等于判断现实世界真相。

## 固定实验

- checkpoint：MoritzLaurer/mDeBERTa-v3-base-mnli-xnli，固定revision `8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c`。
- 只使用safetensors，禁止remote code/pickle，CPU float32、2线程、GPU不可见、本地文件离线加载。
- 权重557652046字节；官方LFS SHA-256 `65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e`，本地及服务器加载前均核对一致。
- 24项预先标注fixture，支持/未知/矛盾各8项，四组：人物归属、引用/假设、版本变化、正负控制。新名字/地点/职业与先前核心失败例不同。人工小规模可行性集，不宣称代表真实分布。
- 金标只用于评分，不进入模型；不截断输入，超过512token直接拒绝实验，不丢限定词；标签映射从配置读取并验证。
- 提前固定高分错误支持阈值0.9，没有为了通过调阈值。

## 结果与拒绝采用的依据

14/24 argmax标签正确；8个entailment全对，8个neutral仅1对，8个contradiction为5对。存在5项错误entailment分数≥0.9：

| 错误推断 | 支持分数 |
| --- | ---: |
| 朋友学篆刻→用户学篆刻 | 0.997 |
| 用户去植物园→角色去植物园 | 0.999 |
| 姐姐辞职、本人没辞职→用户辞职 | 0.997 |
| 姑姑在档案馆工作→用户在档案馆工作 | 0.994 |
| 助手猜会计、用户否认→用户是会计 | 0.995 |

另有小说住址归给用户（0.878）、必要条件误当现实出游（0.875）。在含糊取消上，把未知误判为矛盾；这也不能直接用于自动撤回。模型能处理部分简单否定与显式搬家，但不能用这些通过项掩盖主体与引述错误。

单条前向平均0.170秒，首次预热不计，24条+1预热=25次分类前向，0次生成式模型调用。延迟不包含常驻模型加载和其他服务开销；速度可用不代表语义可用。没有把分数写入用户记录，没有新增默认依赖、模型常驻进程或每轮审查。

## 下载、资源与产物

服务器直连Hugging Face失败，已确认进程退出。改由本机下载固定版本，第一次权重下载在412842902字节处EOF（终端脚本曾打印Downloaded，但没有据此当成功）；curl续传连接超时退出。随后PowerShell Resume续传完成并通过官方哈希，才上传和运行。没有修改服务器网络配置。模型与分词器共约0.58GB，本机.tmp/r93-nli-model及服务器evaluations/r93-nli-model各保留一份供复核，不在生产目录/源码提交范围内。

evaluation/memory_entailment_probe.py、fixtures/memory_entailment_transfer_20260927.json；.tmp/mechanism-r93-nli-{predictions.jsonl,manifest.json}。远端mechanism-r93-nli正常REMOTE_EXIT=0。此失败改变下一步：不采用该NLI模型为记忆准入/撤回/输出纠错的可信门，不能因模型“专用”就绕过人物归属验证。
