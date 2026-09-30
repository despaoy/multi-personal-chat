# r137：操作确认直读回执，冷读通过但暖历史/混合请求仍失败

## 修改

CompiledCharacterContext携带仅status/persisted的操作回执。明确纯删除指令由回执直接产生确认，标记memory_operation与model_invoked=False，API模型标签memory/operation。不让回答模型重新猜测已执行结果，不修改人物prompt。pending/partial/cancelled/failed/no_change不得声称删除成功；成功措辞仅为匹配的长期记忆及关联来源，并明确聊天记录未删除。

有限指令语法仅决定如何呈现已执行回执，不授予删除权限；疑问、否定、引述、连接其他任务的句子退回正常生成。没有按名字、城市写分支。混合任务保持原句交给模型，此路径尚不能保证确认正确（见实测）。

## 本地测试

广回归1619通过、1跳过、2563未选、4既有警告，91.54秒，.tmp/r137-wide.xml。其后补充完整混合任务入模测试，操作响应文件23项通过；确定性回退模型标签4项通过。与广回归重叠不累计。Ruff及定向diff通过。

## 真实模型：18轮HTTP200，不是全通过

使用服务器隔离r137快照，DeepSeek全内部判断接入、无LoRA/无RAG，来源召回开启，0本地生成。生产未改。

冷读原基线9轮：27次新调用（7answer、9writer、9policy、2selection），两个删除确认不再调用回答模型。猫名/住址claims与sources清空，即时确认正确，后续无值召回，否定删除专业保留。冷读追问仍出现“没有保存过”或邀请重述，不是理想删除体验。

新增暖历史/混合9轮：29次新调用（8answer、9writer、9policy、2selection、1semantic）。

- 暖宠物：首次writer候选为“用户养了一只叫松糕的狗”，但准入accepted=0，结构化claim为空、source=1；模型已口头声称记住。删除writer没有候选，回执no_change，确定性回复准确说明未确认删除，**来源未清除**。后续回答再次提及“松糕”，违背不再引用意图。根因包含自然概括准入拒绝、source-only删除无目标、暖历史仍入模，不能仅归因历史。
- 混合住址删除+四季解释：claim/source实际清空，回执erased；四季解释核心正确，但模型仍声称没有可删除的记忆库。**任务没有被吞掉，但操作确认失败**。
- 否定删除：地质学claim保留，后续答对。

证据：.tmp/r137-cold-manifest.json、.tmp/r137-cold-traces.jsonl、.tmp/r137-warm-manifest.json、.tmp/r137-warm-traces.jsonl。新增interactive观测已记录最终prepared回执。所有远端进程结束，SSH退出。

## 下一步

混合请求应将操作确认与独立问题分开执行再组合，避免模型重判操作；不能靠追加复核或吞掉第二任务。删除须覆盖source-only资料，同时保留可审计的授权范围，不能简单清空其他资料。重新审查generic-fact准入的自然概括兼容性。暖历史与删除后的状态表达另需明确可验证机制。RAG充分性、动态上下文消融、失败恢复与生产级鉴权/并发验收仍未完成，目标active。
