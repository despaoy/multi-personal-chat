# 位置记忆真实模型链路回放（r97）

## 范围

服务器现有 `qwen3-8b-instruct-awq`，无 LoRA。显式启用语义记忆诊断路径，调用项目原有记忆客户端（temperature=0、max_tokens=768、thinking=false），实际执行 scheduler、候选解析、SQLite 写入、CharacterMemoryService 和共享 generation。每例使用新的临时 SQLite；embedding 使用运行时本地模型，日志显示 semantic_status=available（无记录例 disabled）。不是生产 HTTP 服务、完整人物上下文、RAG 或生产 PostgreSQL 端到端测试。

八组用例事先固定，预期字段不进入模型输入。每轮写入后显式 flush，随后模拟不携带历史的新会话读出，故能检测持久记忆本身是否正确。此设计不衡量后台队列的真实等待体验。仅在 `/home/boot/lhm/` 下隔离运行，未发布、重启或修改生产数据。

## 首轮结果

共 15 次真实记忆生成、1 次真实回答生成；其余查询走既有字段直接读出。

| 场景 | 真实结果 | 归因 |
|---|---|---|
| 分两轮告知来源地与住址 | 衡阳/舟山均正确 | r96 字段契约在此例有效 |
| 一句告知两类位置 | 两项都没存 | 模型输出 kind=user_fact，不符合已有枚举；不是窗口装不下 |
| 明确搬家 | 仍读湖州，未更新淄博 | 模型给出正确 SUPERSEDE 与目标 ID，但顶层数组被解析成无 memories 的单个对象 |
| 重复同一句住址 | 一条 active，读遵义 | 模型 MERGE 被已有机制转为 NOOP |
| 朋友住址 | 未存朋友住址，最终回答来自抚顺、住址未知 | 模型仍错误归属为用户，本地校验挡住；本人来源地自由改写导致字段直读 unverified，使用一次生成 |
| 临时出差 | 现居地仍黄山，另存柳州出差 | 未覆盖现居地；未验证所有临时旅行表达 |
| 隐含撤回住址 | 错误保留并读咸阳 | 模型提出 ADD other_user_fact，不是 RETRACT；候选被拒后旧 active 未失效 |
| 老家/定居自然表达 | 模型提议两条，最终仅存一条 user_location，回答无来源地/现居地信息 | r98修正归因：昵称规则先把“老家”误认第三方并拒绝；不是此例走到去重才丢失。另有 legacy key 与 typed 字段存在性不一致 |

原始证据：`.tmp/mechanism-r97-results.jsonl`、`mechanism-r97-manifest.json`；服务器 `evaluations/r97-location-replay/`。每轮含完整模型提议、持久行记录、状态、检索诊断、读出和回答。不得将“已拒绝错误候选”当成成功处理用户更正。

## 本轮修复与验证

`memory_llm._extract_json` 支持合法的顶层候选数组，只把容器规范化为 memories 数组，不推断候选含义。所有证据、归属、目标和生命周期校验继续执行。对象缺少 memories 时明确报错；截断数组必须整体解析失败，不能取其中首个完整对象后静默 no_change。空数组与空 memories 都表示明确无候选。没有新增提示词、城市规则、分类器或模型复核。

跨字段独立测试用专业事实，而非仅覆盖搬家样例；初始 8 failed/2 passed，修复后相关 44 passed。其他前轮机制回归 39 passed；新增/修改模块 Ruff 通过。

固定首轮真实输出重放到新的真实 SQLite（不再调用模型）后，三条版本记录中旧湖州为 superseded，新淄博为 active，来源地渭南保留。证据 `.tmp/mechanism-r97-recorded-replay/result.json`。这隔离证明了容器解析的因果，不依赖第二次模型是否输出相同格式。

随后在独立 `mechanism-r97-fixed-source` 快照重新执行该三轮真实模型案例：又产生顶层 JSON 数组，更新正常保存，最终回复“你来自渭南，现在住在淄博”，三次写入生成、零次读出生成。证据 `.tmp/mechanism-r97-fixed-{results.jsonl,manifest.json}`；两次服务器实验均正常退出，SSH 已关闭，无新增常驻推理服务。本轮总计 19 次真实生成。首轮单次后台记忆生成约 5.28–12.22 秒，检索约 0.87–67.46 毫秒；不把后台 flush 的等待当作线上回复延迟，也不声称性能已完成验收。

## 未完成

不自动把 user_fact 猜成某个具体 kind；不通过模糊位置覆盖家乡/住址；不把被拒绝的隐含撤回当成功。语义类型契约、同 key 多候选保留与未知字段的完整性、隐含撤回的处理仍需下一轮修复和独立转移评测。人物自然度与 RAG 尚未因此得到验收。

归因更正（r98）：逐候选检查并对冻结 r97 代码做同输入重放，确认原始自然表达两条中“老家是常德”在 `_NAMED_THIRD_PARTY_PATTERN` 已拒绝；本轮去重修复前后这个原始输出都只接纳台州。改用明确自我归属的独立输入“我老家是汕头，现在定居在中山”，ADD/COEXIST/PENDING三种关系在旧代码均由2条变1条，新代码保留2条，才是去重缺陷的独立证据。详见 `.tmp/mechanism-r98-dedup-audit.json`。不能用修复去重来宣称已经修复“老家”误判。

r99更新：对kind=user_fact且没有旧目标的候选，仅在既有独立提取器能从evidence确认唯一同字段、同值、无条件事实时恢复具体kind，并按原文事实重建content。原始combined_fields输出固定重放后两字段都保存；服务器再次真实生成仍返回user_fact，两字段正确保存、直接读出。证据 `.tmp/mechanism-r99-recorded-replay/result.json` 和 `.tmp/mechanism-r99-results.jsonl`。不是将所有未知类型一律当位置。

同轮temporary_visit真实复测不能记为完整通过：回答黄山住址正确，但模型把出差的valid_from和valid_to均设为调用当前时刻，实际存储后立即过期；因此field_presence中的origin=false没有触发r98的legacy未知分支。这暴露尚未验证的事件有效期语义，不能用表面正确的住址回答掩盖出差记忆丢失。
