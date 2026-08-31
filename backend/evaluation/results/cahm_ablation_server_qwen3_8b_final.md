# CAHM Ablation Results

- Generated: 2026-08-30T09:12:01.561981+00:00
- Gold Set: 60 cases (18 extraction, 42 retrieval)
- Embedding: paraphrase-multilingual-MiniLM-L12-v2
- Context extraction evaluated: True

| Group | Ext P | Ext R | Ext F1 | R@1 | R@5 | MRR | Wrong injection | Avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Baseline | 1.0000 | 0.6667 | 0.8000 | 0.5938 | 0.5938 | 0.5938 | 0.1364 | 0.03 |
| B. Semantic Retrieval | 1.0000 | 0.6667 | 0.8000 | 0.8438 | 1.0000 | 0.9219 | 0.6000 | 9.43 |
| C. Context Extraction | 1.0000 | 0.7333 | 0.8462 | 0.5938 | 0.5938 | 0.5938 | 0.1364 | 0.05 |
| D. Full CAHM | 1.0000 | 0.7333 | 0.8462 | 0.6562 | 0.7188 | 0.6875 | 0.2581 | 8.68 |

## Failure cases

### A. Baseline extraction (5 total)

- `e11` 我平时不碰含咖啡因的东西。 — gold=['semantic:dislike:含咖啡因的东西'], predicted=[]
- `e12` 最近主要在复现点云模型。 — gold=['semantic:goal:复现点云模型'], predicted=[]
- `e13` 还是上次那个方向。 — gold=['semantic:goal:点云补全'], predicted=[]
- `e14` 补全方向暂时不考虑了，我最近改做点云识别。 — gold=['semantic:goal:点云识别'], predicted=[]
- `e15` 不是完全讨厌咖啡，只是不喜欢太苦的。 — gold=['semantic:dislike:太苦的'], predicted=[]

### A. Baseline retrieval (16 total)

- `r01` 我保研准备得怎么样？ — gold=['m1'], selected=[], wrong=[]
- `r02` 推免准备到哪一步了？ — gold=['m1'], selected=[], wrong=[]
- `r05` 我当前在做什么研究工作？ — gold=['m1'], selected=[], wrong=[]
- `r06` 最近的科研方向进展如何？ — gold=['m1'], selected=[], wrong=[]
- `r08` 我在哪座城市生活？ — gold=['m1'], selected=[], wrong=[]
- `r09` 我就职于哪家公司？ — gold=['m1'], selected=[], wrong=[]
- `r10` 我读的是什么方向？ — gold=['m1'], selected=[], wrong=[]
- `r11` 点云识别最近进展如何？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r19` 我最近准备的事情怎么样？ — gold=['m2'], selected=[], wrong=[]
- `r20` 研究任务有进展吗？ — gold=['m2'], selected=[], wrong=[]

### B. Semantic Retrieval extraction (5 total)

- `e11` 我平时不碰含咖啡因的东西。 — gold=['semantic:dislike:含咖啡因的东西'], predicted=[]
- `e12` 最近主要在复现点云模型。 — gold=['semantic:goal:复现点云模型'], predicted=[]
- `e13` 还是上次那个方向。 — gold=['semantic:goal:点云补全'], predicted=[]
- `e14` 补全方向暂时不考虑了，我最近改做点云识别。 — gold=['semantic:goal:点云识别'], predicted=[]
- `e15` 不是完全讨厌咖啡，只是不喜欢太苦的。 — gold=['semantic:dislike:太苦的'], predicted=[]

### B. Semantic Retrieval retrieval (42 total)

- `r01` 我保研准备得怎么样？ — gold=['m1'], selected=['m2', 'm1'], wrong=['m2']
- `r02` 推免准备到哪一步了？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r03` 推荐饮料要避开什么？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r04` 我的饮品偏好是什么？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r05` 我当前在做什么研究工作？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r06` 最近的科研方向进展如何？ — gold=['m1'], selected=['m2', 'm1'], wrong=['m2']
- `r07` 升学申请准备得怎样？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r08` 我在哪座城市生活？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r09` 我就职于哪家公司？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r10` 我读的是什么方向？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']

### C. Context Extraction extraction (4 total)

- `e02` 我的专业是计算机。 — gold=['key:user_major'], predicted=[]
- `e09` 下次一定带你去看展。 — gold=['key:promise_带你去看展'], predicted=[]
- `e13` 还是上次那个方向。 — gold=['semantic:goal:点云补全'], predicted=[]
- `e15` 不是完全讨厌咖啡，只是不喜欢太苦的。 — gold=['semantic:dislike:太苦的'], predicted=[]

### C. Context Extraction retrieval (16 total)

- `r01` 我保研准备得怎么样？ — gold=['m1'], selected=[], wrong=[]
- `r02` 推免准备到哪一步了？ — gold=['m1'], selected=[], wrong=[]
- `r05` 我当前在做什么研究工作？ — gold=['m1'], selected=[], wrong=[]
- `r06` 最近的科研方向进展如何？ — gold=['m1'], selected=[], wrong=[]
- `r08` 我在哪座城市生活？ — gold=['m1'], selected=[], wrong=[]
- `r09` 我就职于哪家公司？ — gold=['m1'], selected=[], wrong=[]
- `r10` 我读的是什么方向？ — gold=['m1'], selected=[], wrong=[]
- `r11` 点云识别最近进展如何？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r19` 我最近准备的事情怎么样？ — gold=['m2'], selected=[], wrong=[]
- `r20` 研究任务有进展吗？ — gold=['m2'], selected=[], wrong=[]

### D. Full CAHM extraction (4 total)

- `e02` 我的专业是计算机。 — gold=['key:user_major'], predicted=[]
- `e09` 下次一定带你去看展。 — gold=['key:promise_带你去看展'], predicted=[]
- `e13` 还是上次那个方向。 — gold=['semantic:goal:点云补全'], predicted=[]
- `e15` 不是完全讨厌咖啡，只是不喜欢太苦的。 — gold=['semantic:dislike:太苦的'], predicted=[]

### D. Full CAHM retrieval (16 total)

- `r01` 我保研准备得怎么样？ — gold=['m1'], selected=[], wrong=[]
- `r05` 我当前在做什么研究工作？ — gold=['m1'], selected=[], wrong=[]
- `r06` 最近的科研方向进展如何？ — gold=['m1'], selected=['m2'], wrong=['m2']
- `r08` 我在哪座城市生活？ — gold=['m1'], selected=[], wrong=[]
- `r09` 我就职于哪家公司？ — gold=['m1'], selected=[], wrong=[]
- `r10` 我读的是什么方向？ — gold=['m1'], selected=[], wrong=[]
- `r11` 点云识别最近进展如何？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r12` 拿铁这个喜好还记得吗？ — gold=['m1'], selected=['m1', 'm2'], wrong=['m2']
- `r19` 我最近准备的事情怎么样？ — gold=['m2'], selected=[], wrong=[]
- `r20` 研究任务有进展吗？ — gold=['m2'], selected=[], wrong=[]
