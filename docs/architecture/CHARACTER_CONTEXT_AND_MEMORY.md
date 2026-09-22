# 人物上下文与长期记忆

文档核对：2026-09-11，依据本地工作区。记忆写入召回包含核对前已有的未提交代码变更。

## 模块职责

| 文件（相对backend） | 职责 |
| --- | --- |
| services/character_context.py | prepare_turn组装上下文，complete_turn提交记忆与关系变化 |
| character/profile_registry.py | 角色画像及LoRA到画像的映射 |
| character/situation_analyzer.py | 规则识别情境、意图、情绪和边界 |
| character/decision_policy.py | 生成当前轮回复策略 |
| character/semantic_state_estimator.py、semantic_review_adapter.py | 可选模型语义复核，闭集输出与失败回退 |
| character/context_builder.py | 编译角色上下文 |
| character/memory_extractor.py | 高精度规则记忆候选与写入门禁 |
| character/memory_llm.py | 后台召回旧记忆、提出版本操作并本地校验 |
| character/memory_service.py | 为回复检索相关记忆 |
| repositories/character_memory.py | 当前作用域的记忆持久化接口 |

## 动态上下文

稳定人物画像规定身份与行为边界。动态状态根据当前消息和关系状态产生；普通单意图轮不再一律输出机械的“情景/意图/语气/行动/避免”模板。复杂否定、反讽、多意图和指代可触发可选语义复核。

DYNAMIC_CONTEXT_SEMANTIC_REVIEW_ENABLED默认关闭；启用后仍受超时、闭集标签及数值校验约束，失败回退规则结果。角色身份不是根据用户提到的人名自动改变；固定角色部署保持自动LoRA路由关闭。

画像和行为策略属于系统指令。检索出的RAG与用户记忆进入不可信参考边界；角色历史保留消息角色结构，不能简单说所有历史都被改成user消息。

## 写入：先找相关旧记忆，再判断版本关系

1. 检查真实用户来源、拒绝记忆意愿和敏感信息等门禁。
2. 规则提取姓名、偏好、专业、位置、目标等候选，作为rule_hints；不要求每条消息都产生记忆。
3. 显式记住、纠错或遗忘走hot调度；普通陈述空闲或达到批量阈值后提交。批量是后台调度策略，不等于把四句自动合成一条事实。
4. 在当前作用域读取全部active记录（limit=None），按词面与embedding余弦相似度召回；规则memory_key精确匹配和本轮反馈目标优先，词面与语义排名用RRF融合。embedding失败时降级为词面路径。
5. 去重并选择最多10条已有记忆；LLM只在这些目标白名单内判断ADD、MERGE、SUPERSEDE、COEXIST、PENDING、RETRACT、NOOP或ERASE。
6. 本地检查JSON、证据来自当前用户原文、目标白名单、置信度、时间、主体和作用域授权后持久化。LLM不能直接任意修改数据库。

历史上下文最多4条，每条500字符；当前消息默认上限2000字符；旧记忆正文每条最多120字符，携带ID、key、类型、状态和有效时间；输出最多4条候选。历史assistant消息可帮助消歧，但不能作为用户事实的原文证据。

这些是条数和字符限制，不是对完整请求逐token预算。当前写入搜索会加载并向量化全部可见active记忆，成本随数据量增长；Top-K只限制交给LLM的候选数，不限制前面的扫描和embedding成本。大规模使用需要持久化向量索引、增量更新及token预算等进一步工作。

## 读取：为当前回答选择记忆

memory_service.load_relevant_memories先读取配置限制内的候选，按生命周期、有效时间和主体过滤，再通过查询扩展、词面、语义、意图、重要度与新近度排序，使用RRF融合并选出少量相关记忆。明确询问过去时可走历史版本查询。

读取链路仍有候选读取上限；不能宣称每次回复都会对数据库所有旧记忆全局召回。未被选入上下文不等于数据库记录被覆盖或删除。

## 版本、作用域和交付

记忆区分conversation、user_character和user_global。跨会话/角色晋升须有用户明确授权；平台、适配器和用户身份继续隔离。替代记录保留版本关系，明确删除另走ERASE。

AstrBot入口只有首次成功发送确认后才完成角色状态更新；重复确认不重复更新。直接Web对话不具有平台送达确认步骤。服务prepare_turn/complete_turn是业务边界，实际完成时机由入口决定。

## 验证边界

相关测试位于test_character_memory_llm.py、test_character_memory_versioning.py、test_balanced_memory_retrieval.py和test_character_context.py。它们与真实LLM的召回、版本判断、延迟和大规模记忆评测分别记录。工作区存在实现和测试文件不等于本轮已经运行通过。
