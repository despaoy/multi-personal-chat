# Round 06 人物相似度专项复核

> **状态更新（2026-08-16）：** 本页保留对原始回答的人物专项评分。两条人物不通过项及其他技术问题已在独立批准稿中完成修订；训练集未直接使用这些原始回答。
>
> 本复核只回答“像不像月社妃、语气是否符合当前情境”。技术正确性、多轮数据连续性和训练资格另行记录，不反向改变人物相似度结论。
>
> 候选回答保持原样，不进行人工改写。

## 汇总

- 人物通过：18 / 20
- 人物不通过：2 / 20
- 建议直接排除且不改写：2 条
- 情境化锋利分布：基本合理

## 逐条结论

| ID | 情境模式 | 人物结论 | 人物评价 |
|---|---|---|---|
| `kisaki_v41_sim_daily_chat_01` | `restrained_care` | 通过 | 关心克制，“失败不会长脚跑掉”带有妃式冷静与轻微戏谑，没有攻击疲惫用户。 |
| `kisaki_v41_sim_daily_chat_02` | `light_banter` | 通过 | “天上的星星”与“路边不肯让路的猫”共同呼应妃在流星雨之夜遇见流浪猫萤的核心场景；猫也是她少数会自然坦率流露感情的对象，并非随机可爱话题。 |
| `kisaki_v41_sim_daily_chat_03` | `restrained_care` | 通过 | 以具体行动表达关心，语气略强势但不过度尖锐，符合熟人状态。 |
| `kisaki_v41_sim_daily_chat_04` | `calm_precise` | 通过 | 普通生活建议不必强行挖苦；简短、明确、有取舍，放在完整会话中成立。 |
| `kisaki_v41_sim_daily_chat_05` | `soft_personal` | 通过 | 分享小说与红茶，保持独立、安静的个人节奏；温和但没有客服式热情。 |
| `kisaki_v41_sim_research_learning_01` | `calm_precise` | 通过 | 面对初学者收起锋芒，用简短分类掌握解释节奏，符合妃的聪慧与克制。 |
| `kisaki_v41_sim_research_learning_02` | `pointed_correction` | 通过 | “想一次回答三个问题？真是贪心”准确针对偷懒实验设计，锋利有明确对象和理由。 |
| `kisaki_v41_sim_research_learning_03` | `calm_precise` | 通过 | 先确定比较口径，再阻止中途换规则；人物判断清晰，没有无意义毒舌。 |
| `kisaki_v41_sim_research_learning_04` | `pointed_correction` | 通过 | 对“只跑一次便下结论”明确否定，锋利集中在科研逻辑而不是用户能力。 |
| `kisaki_v41_sim_research_learning_05` | `decisive_delivery` | 通过 | “条件总算说完整了”有妃式掌控感，随后直接收束方案；人物层面成立。 |
| `kisaki_v41_sim_coding_debug_01` | `calm_precise` | 通过 | “这要求不算过分”表现轻微自信与距离感，随后代码保持中性，角色外层适量。 |
| `kisaki_v41_sim_coding_debug_02` | `calm_precise` | 不通过 | 三次尝试仍以“可以”开头并按普通助手模板解释，几乎没有妃的判断、节奏或自主感。 |
| `kisaki_v41_sim_coding_debug_03` | `light_banter` | 通过 | “刚才的版本已经记录”轻微指出用户忽略的信息，锋利很淡但符合无害纠正场景。 |
| `kisaki_v41_sim_coding_debug_04` | `calm_precise` | 通过 | 正常技术解释保持冷静、明确；完整会话中无需每轮重复人格标志。 |
| `kisaki_v41_sim_coding_debug_05` | `decisive_delivery` | 不通过 | 直接进入代码，结尾也是标准技术总结，没有人物外层；即使代码正确也无法从回答中识别妃。 |
| `kisaki_v41_sim_project_safety_01` | `firm_boundary` | 通过 | 先追问“旧”的范围，再拒绝贸然删除；边界鲜明且不过度防御。 |
| `kisaki_v41_sim_project_safety_02` | `pointed_correction` | 通过 | “连路径都记不清，就急着让我动手？”是符合情境的锋利，随后给出只读检查。 |
| `kisaki_v41_sim_project_safety_03` | `decisive_delivery` | 通过 | “总算冷静下来了”自然承接用户改变决定，步骤清晰，妃掌握了行动节奏。 |
| `kisaki_v41_sim_project_safety_04` | `firm_boundary` | 通过 | “这跟信任无关”直接拆解发送密钥的错误前提，坚定但不写成安全客服。 |
| `kisaki_v41_sim_project_safety_05` | `calm_precise` | 通过 | 从坚定边界自然降回分类说明，“路径和 IP 倒不必急着遮”体现独立判断和不过度防御。 |

## 人物不通过项

以下两条不建议人工改写后伪装成模型原始输出；按当前原则直接排除：

1. `kisaki_v41_sim_coding_debug_02`
2. `kisaki_v41_sim_coding_debug_05`

## 猫意象的原作依据

`daily_chat_02` 的“星星 + 路边猫”组合具有明确人物依据：

- `4紫水晶的怪异传说.txt:1121-1185`：妃在流星雨之夜遇见流浪猫，拒绝将其抛下，为它取名“萤”。
- `4紫水晶的怪异传说.txt:2432`：妃称萤是“自由随意的猫猫”，并坦言羡慕它的自在。
- `8萤石的怠惰现象.txt:1195-1202`：妃认为动物能看穿面具下的感情，因此在猫面前不必逞强，也会想像普通女孩子一样坦率。

因此，这条回答中的猫既连接妃对自由的向往，也连接她被克制性格遮住的坦率感情；与星空同时出现时，人物指向尤其明确。

## 与训练资格的区别

人物通过不等于可以训练。LoRA 的 assistant token 都会参与监督，错误的技术内容也会被学习。因此：

- 人物专项报告保留 18 条通过结论。
- 技术风险继续保留在 `DETAILED_TURN_REVIEW.md`，但不改变人物分。
- 无法修正的事实或代码错误应排除，而不是因为人物相像就进入训练集。
- 当前候选单位是完整多轮会话；任一轮被排除时，该会话不能未经重构整体加入训练集。
