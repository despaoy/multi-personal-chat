# 面向人物还原的 LoRA、DPO 与 PPO：研究与项目方案

检索与代码阅读日期：2026-09-23。本文是研究建议，不是训练完成报告。论文依据为本次访问的论文摘要页；项目依据为公开 README、部分源码和本地实现。未复现外部实验，也不把作者案例演示等同于独立效果验证。超参数、数据量和实验门槛中标为建议的内容，均需本项目验证。

## 1. 结论

推荐路线：修复人物 SFT 数据与上下文 → 建立可靠 LoRA 基线 → 面向具体错误的 DPO → 在独立场景验证 → 仅在奖励可靠、确有多轮优化需求时加入 PPO。

LoRA 是参数更新方式；SFT、DPO、PPO 是训练目标/算法。可以用 LoRA 做 SFT，也可以用 LoRA 做 DPO 或 PPO。DPO 与 PPO 不必全部串起来：先从同一合格 SFT checkpoint 分别比较 SFT、SFT+DPO、SFT+PPO，最后才判断 DPO→PPO 是否值得。

“更像人物”至少包含：语言节奏、价值判断、关系差异、情绪反应、知识边界、跨轮连续性。口癖和人物名字只是其中很小的一部分。

## 2. 本地项目已具备什么、问题在哪里

| 观察 | 本地依据 | 对下一步的影响 |
|---|---|---|
| 基座为 Qwen3-8B，已有 LoRA SFT、DPO/ORPO 入口 | `backend/training/trainer.py`、`preference_trainer.py` | DPO 无需从零搭建；PPO 需要独立训练流程 |
| 偏好后端目前只接受 dpo/orpo | `backend/training/preference_compat.py` | 不能只加一个 `method=ppo` 就完成 PPO |
| 当前无通过门禁的正式 adapter | `README.md`、`docs/research/KISAKI_EXPERIMENT_INDEX.md` | 不应默认最后一个 SFT checkpoint 就是合格起点 |
| 旧 recovery checkpoint 越训越有口吻，但事实编造、关系误判和场景串扰增加 | `review_packets/kisaki_v4/20_R1V4_E1_CHECKPOINT_SCREEN_AND_METHOD_REDESIGN/README.md` | 首先修复训练信号；增加 rank 或优化步骤不解决标签错配 |
| V4 926 条记录中，Codex 长模拟占实际监督字符 82.26%，原作占 14.00% | `backend/data/character_dialogues/experiments/v5_candidate/README.md` | 按实际监督目标及 token/字符暴露审计，不能只按记录数比较；字符占比不等于精确梯度占比 |
| 原作台词中位长度 17 字、84.2% 不超过 30 字 | `docs/research/KISAKI_CHARACTER_PROFILE.md` 的本地统计 | 日常训练应保留短句与留白；解释、冲突和情绪高潮允许长答，不能统一奖励短回复 |
| R4 已有人工审核、冻结和 80/20 切分脚本 | `scripts/prepare_kisaki_dpo_v3.py`、`scripts/lab-run-kisaki-r4-dpo.sh` | 总览仍记录等待至少 100 条人工批准偏好对；这是本地试验入口要求，不是论文证明的数据充足阈值 |
| 规则人物评分主要靠关键词、长度和简单否定共现 | `backend/evaluation/persona_metrics.py` | 可用于冒烟检查，不适合直接当人物 PPO 奖励 |
| 已有证据条件 SFT/DPO 格式、来源分组隔离、上下文长度校验 | `backend/training/evidence_dataset.py` | 优先复用；不必重新发明所有数据协议 |

### DPO 训练前需要补齐的检查

1. **明确参考策略。** 当前代码加载可训练 SFT adapter，构造 DPOTrainer 时没有显式指定 `ref_model` 或参考 adapter。TRL v0.13 文档说明，可采用“同一个 SFT adapter 加载两份，训练一份、冻结一份”；不当配置可能关闭 adapter，以基座作参考。本次查看的 TRL main 源码已包含自动复制预训练 adapter 为 `ref` 的逻辑。因此这是**版本相关的复现风险，不是已经证明所有运行都引用错误**。锁定 TRL/PEFT/Transformers 版本，并验证初始 policy/reference 输出一致、训练后 reference 不变；如需对照基座参考，单列实验。
2. **adapter 路径必须失败即停。** 当前指定路径不存在时可进入新建 LoRA 分支，容易把“接续 SFT”跑成“从基座开始”。
3. **增加独立验证。** 当前训练器不传 `eval_dataset`，结果也明确未计算 held-out preference win rate。补偏好集评分与真实生成盲评；chosen 的 log-prob 优势不等于实际人物胜率。
4. **按场景来源切分。** R4 旧冻结脚本去除相同规范化 prompt 后逐行随机切分，不能隔离同一原作场景的近义改写。复用 evidence 导出中的 source-group/source-id 隔离，再做文本近重复审计。
5. **避免上下文截断。** 默认 `max_length=512`、`max_prompt_length=256` 对人物画像+证据+多轮历史可能不足。先统计实际 tokenizer 长度，再定预算；不能把关系、事实删掉后要求模型答对。
6. **统一训练/推理协议。** 人物画像、可信对话者身份、场景状态、证据格式、chat template、Qwen3 思考模式和 EOS 均应一致。仅对目标人物的回答计算相应损失。

TRL 参考：[v0.13 DPO/PEFT 文档](https://huggingface.co/docs/trl/v0.13.0/en/dpo_trainer)、[当前 DPO 文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)、[本次查看的 main 源码](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)。main 为浮动版本，正式实验应记录 commit。

## 3. 优先阅读的论文

| 论文 | 核心内容 | 对本项目的用途与局限 |
|---|---|---|
| [Direct Preference Optimization (2023)](https://arxiv.org/abs/2305.18290) | 用偏好对直接优化策略，无需训练独立奖励模型和在线 RL 采样 | 适合在同一情境下纠正“看起来有口吻、实则关系/事实错误”的回复；通用对齐结果不能直接证明月社妃效果 |
| [Proximal Policy Optimization Algorithms (2017)](https://arxiv.org/abs/1707.06347) | 交互采样、策略概率比和受约束的更新 | 理解 PPO 算法；原始实验不是语言角色扮演实验 |
| [Training language models to follow instructions with human feedback (2022)](https://arxiv.org/abs/2203.02155) | SFT、偏好标注、奖励模型、语言模型 RLHF 的完整流程 | 理解如何将 PPO 用于语言模型；其通用助人偏好不能直接作为角色忠实度奖励 |
| [RoleLLM (2023)](https://arxiv.org/abs/2310.00746) | 画像构建、情境知识提取、风格模仿、角色条件微调 | 借鉴数据流水线，把知识与表达分开；避免批量生成同质的通用助手回答 |
| [Character-LLM (2023)](https://arxiv.org/abs/2310.10158) | 人物画像、经历和情绪状态的训练 | 把稳定性格与当前情绪区分开；合成经历仍需事实依据 |
| [CoSER (2025, ICML)](https://arxiv.org/abs/2502.09082) | 真实文学对话、场景设置、人物经历与 given-circumstance acting | 最贴近当前瓶颈：训练必须包含“对谁、何时、发生过什么、现在想做什么”；结果不构成 DPO/PPO 优越性证明 |
| [Character is Destiny / LIFECHOICE (2024)](https://arxiv.org/abs/2404.12138) | 人物驱动决策评测；CHARMAP 人物记忆检索 | 检验她在冲突中如何选择，而非只问她有什么性格；可建本地决策情境集 |
| [InCharacter (2023/ACL 2024)](https://arxiv.org/abs/2310.17976) | 通过心理访谈与量表评测 personality fidelity | 补充性格测量；量表只覆盖一部分人物特征，仍需原作场景验证 |
| [CharacterEval (2024)](https://arxiv.org/abs/2401.01275) | 中文多轮角色对话、四维度十三指标、人物奖励模型 | 借鉴人工标注准则和 CharacterRM；不能假设该奖励模型直接懂月社妃及项目事实边界 |
| [RoleRAG (2025)](https://arxiv.org/abs/2505.18541) | 实体消歧、图检索、人物认知边界 | 让事实检索考虑人物当前知情范围，避免把整部作品全知视角灌给角色 |
| [ERABAL (2024)](https://arxiv.org/abs/2409.14710) | 针对与人物属性相近但越界的问题做边界学习 | 借鉴“很像她可能知道、实际未有依据”的困难负例；作为补充阅读 |

建议阅读顺序：CoSER → DPO → CharacterEval → LIFECHOICE → PPO/InstructGPT。重点是明确人物目标与可验证训练信号，再选择优化器。

## 4. 开源项目与实践

| 项目 | 已核实内容 | 推荐借鉴 |
|---|---|---|
| [Hugging Face TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer) | DPO 数据格式、PEFT 集成、参考策略与多种偏好损失 | 当前仓库已用 TRL，继续沿用最省改造；锁定版本后实现参考一致性和验证集评估 |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | README 明确支持 SFT、reward modeling、PPO、DPO、LoRA/QLoRA 和 Qwen3 | 作为标准训练流程对照；不必迁移整套业务系统；框架列出功能不等于所有模型/量化/算法组合都兼容 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | SFT、DPO、奖励模型、PPO、多轮 agent、自定义奖励、LoRA/QLoRA 等训练能力 | 如进入 PPO，优先复用 rollout/reward/critic 工程；先做本机显存与吞吐试验，不根据功能列表假定单卡足够 |
| [RoleLLM-public](https://github.com/InteractiveNLP-Team/RoleLLM-public) | RoleBench 数据、角色画像和知识/风格构建框架 | 借鉴角色化数据结构及评测设计，不将其他人物台词直接混入单人物训练 |
| [CharacterEval / CharacterRM](https://github.com/morecry/CharacterEval) | 中文多轮数据、标注示例、已发布奖励训练数据、BaichuanCharRM 推理脚本 | 借鉴评分准则；先拿本地人工标注对校准，观察是否偏爱长答、姓名堆砌和过度亲密 |
| [ZhangWuji-LLM-RolePlay](https://github.com/ZHAOoops/ZhangWuji-LLM-RolePlay) | Qwen2.5-7B+LoRA 的 SFT→DPO 案例，存在 DPO 数据、训练和评估文件；README 展示亲属/身份混淆纠正案例 | 很贴近“关系事实难以靠 SFT 修好”的情况：用模型真实错答作 rejected、人工核验答案作 chosen。README 的“彻底解决”等表述只是作者自述，不能当作严格泛化证据 |

进一步查看张无忌案例源码：`src/training/train_dpo.py` 使用旧版 TRL 0.8.6 接口、4-bit 基座、可训练 SFT adapter、`beta=0.1`、`lr=1e-6`、50 steps，且 `ref_model=None`；`src/evaluation/evaluate_model.py` 主要计算 ROUGE-L 与语义相似度。这些指标不能独立证明人物决策、关系边界或多轮忠实度。应借鉴其困难负例思路，不直接复制旧 API、固定 token ID 或其效果结论。

本次没有找到并核实一个足以证明“对本项目这种单人物少量语料，PPO 必然比 DPO 更好”的直接实践依据。因此 PPO 定位为待检验分支。

## 5. 怎样让月社妃更像月社妃

### 5.1 给训练提供完整情境

每条训练输入包括：稳定人物画像、对话者可信身份与关系、当前场景、时间/剧情进度、当前目标、已知证据、对话历史、尚未完成的约定。关系不能由用户一句“我是琉璃”直接改写；采用系统已确认状态。

相同问题分别放到陌生人、熟人、重要亲人，以及平静、冲突、脆弱的情境中。期望表达可以不同，但事实和人物价值取向应一致。稳定人物状态与瞬时情绪分离，避免把“别扭”学成每次都怀疑、每次都挖苦。

只在适合的场景表现元叙事特征。技术问答中的文件大小不应被解释为“魔法浓度”；涉及重要人物也不应自动补出疾病、探望或共同经历。

### 5.2 分配各组件责任

- LoRA/SFT：语言节奏、稳定判断倾向、关系条件表达、自然边界回应。
- DPO：同一情境下，辨别哪种表达、选择和证据使用更合适。
- 角色 RAG：人物关系、剧情事件、时间线、角色已知事实及原文证据。
- 会话记忆：本轮和历史约定、用户明确提供的信息、物品/时间/地点状态。
- PPO（可选）：在可信评分下改善多轮互动轨迹，例如维持承诺、处理冲突后恢复关系。

“记住推理小说、十分钟和下午喝茶”需要可靠状态输入。强化学习不能替代一个根本未向模型提供信息的记忆系统。

### 5.3 构造有辨别力的偏好对

不是“优美长答 > 粗糙短答”，而是“条件相同、尽量只差一个关键错误”。重点类别：

| 类别 | chosen | rejected |
|---|---|---|
| 关系距离 | 对陌生人保持礼貌边界，对亲近者表达克制关心 | 未经建立关系便称呼哥哥、暧昧或分享私密经历 |
| 人物选择 | 在价值冲突中体现有原作依据的取舍 | 通用助手式一味迎合、过度建议或机械说教 |
| 情绪与表达 | 符合当时关系和强度的反应 | 每次都讽刺、笑声堆砌、情绪突然升级 |
| 事实边界 | 使用给定证据；没有依据时自然承认未知 | 编造童年、疾病、约定、物品来源 |
| 多轮连续性 | 保留已给出的物品、时间、地点和边界 | 将推理小说换成笔记本，或虚构上次发生过什么 |
| 知情/剧情边界 | 区分人物现在知道的事与未来剧情 | 全知视角剧透、跨剧情阶段拼接设定 |
| 日常自然度 | 适当简短、具体、允许留白 | 口癖正确但像教程、心理咨询或百科讲解 |

建议每个情境从当前模型采样 3–4 个候选，保留真实易犯错误，再人工修订和审核 chosen。不应只把弱模型输出当负例、强模型输出当正例，否则模型可能只学到长度和行文格式差异。

优先包含“风格很像但事实错”和“事实正确且不用口癖也很像”的困难对。对事实或关系硬错误优先判负，再比较自然度。实质平局、两边都差、证据不足的对丢弃或送复审，不强行贴二元标签。

示意（新构造教学样例，不是原作台词或已审核训练数据）：已确认与熟人约定十分钟后归还推理小说，用户问“我先走，可以吗？”正例应保留十分钟和推理小说，并以合适关系语气回应；负例可以同样流畅、同样有口吻，但把物品说成笔记本。这一对训练的是情境连续性，而非“哪个更华丽”。

单轮 DPO 中，history 是输入，chosen/rejected 是当前回复。它能改善给定历史下的回复，不会自动解决整段对话长期信用分配。需要多轮轨迹数据和评测来验证迁移。

## 6. DPO 与 PPO 的实际区别

| 项目 | DPO | PPO（语言模型常见实现） |
|---|---|---|
| 输入信号 | 固定 prompt、chosen、rejected | 当前策略采样回答/轨迹，再获得奖励 |
| 是否需要独立奖励模型 | 不需要 | 需要奖励信号；可来自奖励模型、规则或外部评分器，并非必须自训一个模型 |
| 常见组件 | policy + 冻结 reference，可用共享基座/双 adapter 降低开销 | actor、reference、reward、critic/value；实现可以共享或卸载部分组件 |
| 适合当前问题 | 有明确对比的关系、事实、风格、决策纠错 | 可靠可评分的在线分布与多轮轨迹优化 |
| 主要风险 | 偏好噪声、场景泄漏、长度偏差、参考策略配错、偏离事实 | 奖励投机、critic 不稳、KL 失控、仿真用户偏差、采样成本 |
| 优先级 | 修复 SFT 后首选 | 有奖励校准证据后再做 |

DPO 优化相对于参考模型的 chosen/rejected 对数概率差。它不是简单把 rejected 从模型里“删除”，也不保证提升所有人格维度。beta 是参考约束相关参数，其实际效果还与损失、学习率、训练步数有关，应实测输出偏移。

PPO 的典型目标是提高角色奖励，同时用 KL 惩罚约束相对 reference 的偏移。PPO clipping 约束新旧策略的更新，reference KL 约束对初始行为的漂移，两者不同。

### PPO 奖励先验证，再优化

建议用独立分项评分：人物决策、关系恰当、情绪一致、事实依据、状态连续、语言自然、任务回应质量。权重先由人工偏好校准，不先拍定一个“最优配比”。口癖数量、角色名出现次数不能作为核心奖励。

事实捏造、关系越界、严重状态矛盾等设硬约束或总奖励上限，避免大量风格加分抵消。然后测试评分器是否会奖励：长篇空话、重复口癖、无条件亲密、始终拒答、讨好用户、透露它如何被评分。

可先用有固定准则的强模型评分器构造候选排序，并抽样人工复核。若只有模型反馈，应如实标记 AI feedback；若进入自训人物奖励模型，训练集和最终评测集必须分离。不要把同一个裁判的训练奖励当作最终效果证据。

在同一批独立场景上，用奖励做 best-of-N 候选排序是一个便宜的先导实验：如果排序都无法稳定选出更好的回答，暂不使用它训练 PPO。排序有效也不是 PPO 必然成功的保证。

多轮 PPO 起步可用少量固定 3–5 轮情境，明确目标人物生成 token、用户模拟 token 和环境状态的边界，只更新目标人物输出；模拟用户风格要多样，并保留人工交互复核。

## 7. 分阶段、可比较的实验

以下规模为建议起点，不是统计充分性或成功保证。

1. **先建立可用 SFT。** V5 source-balanced 候选继续完成审核；减少通用长模拟的监督暴露，保留原作关系和语言节奏。沿用现有低强度 recovery 思路，按生成质量而非仅 eval loss 选 checkpoint。若 SFT 仍输给 prompt-only，保留负结果并继续修数据。
2. **先导偏好集。** 首批 100 条可验证标注与训练流程；若覆盖和一致性合格，再扩到约 300–500 条不同情境的审核偏好对。按场景/来源成组切分 train/dev；冻结 Gold v3 保持最终用途，不能读题后生成近似训练数据。
3. **DPO 最小实验。** 从同一合格 SFT 出发，参考策略也冻结为该 SFT。先沿用 R4 的 `lr=5e-7, beta=0.1, epochs=1`，继承 SFT adapter 的 rank，不同时改多项；实际步骤上限和序列预算由数据长度与 dev 表现决定。保存早期 checkpoint，以防过度对齐。
4. **统一对照。** A=prompt+固定 RAG/记忆；B=相同系统+合格 SFT；C=B+DPO。除训练方法外，人物上下文、检索证据、解码参数和上下文长度固定。失败归因到输入不足、检索错误或生成选择，不把全部提升归于 LoRA。
5. **PPO 分支。** 奖励通过人工校准后，D=从同一个 B 开始 PPO，reference 同 B。与 C 比较性能、GPU 时间、采样量和标注成本。多轮实验单列；若再做 C→PPO，作为 E 单独报告，并明确 reference 是 B 还是 C。
6. **确认与泛化。** 在未参与训练和 checkpoint 选择的新场景上盲评。先一个 seed 做 pilot；仅为基线与胜出方案补另外两个 seed。记录 win/tie/loss 和按场景聚类的置信区间，不把一批少量题的胜率直接写成普遍提升。

评测至少分别报告：人物自然度/决策胜率、严重事实错误数、关系越界数、跨轮状态保持率、重复和过度口癖率、回复长度分布、通用任务能力回退。人物收益不能掩盖明显新增的事实/关系严重错误。

要检查领域外新场景，以及“刻意诱导更亲密”“引用错误原作事实”“提供相似但未发生经历”“长历史后再问约定”等挑战。对多个角色还需测试 adapter 路由和记忆隔离，防止人物之间串扰。

## 8. 推荐工程顺序

优先改 `preference_trainer.py` 的版本/参考策略契约、adapter 路径校验、独立验证和生成评估；扩展偏好元数据记录失败类型、关系、剧情位置、证据来源和分组 ID；复用 evidence 数据模块隔离来源。第一阶段的可交付目标应为“合格 SFT + 可解释偏好集 + DPO 对照结果”。

PPO 阶段另建人物 reward 校准集、rollout 数据协议与训练入口，复用成熟框架。不要复用关键词人物评分器充当 PPO 的主要奖励，也不要为了同时包含两种算法而直接把 DPO 与 PPO 连跑。

本次只新增这份研究文档；没有改动训练代码、审核状态、冻结数据或启动 GPU 训练。
