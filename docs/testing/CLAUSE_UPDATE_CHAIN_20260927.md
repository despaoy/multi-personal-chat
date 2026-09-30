# r130：分句更新修复及下一层真实故障

上一目标轮完成目标创建/验收文件，属于进展。本轮根据现有代码与真实trace重新定位，不信任历史通过结论。用户进一步要求三模块每个旧环节重新检查，已补充完整审计矩阵至 STRONG_MODEL_CHAIN_ACCEPTANCE_20260927.md。

## 原因与修改

memory_extractor._scoped_simple_memories 已逐分句解析、建立证据span，但最终只返回整句解析命中的项目。带起止锚点的谓词（如已经搬家）在单个分句匹配，放入多分句整句就消失。随后 memory_llm 的位置字段推断退回 user_location，与旧 user_residence 目标不符，拒绝 SUPERSEDE。

现在从已解析的分句中补回整句漏掉的字段，仅补同字段唯一候选；不覆盖整句已有字段，不任意选择相互冲突的分句，条件句仍保留整句处理，原证据不改写。不增加城市或姓名特判、不修改人物提示词、不增加模型复核。

新增11项迁移/反例：不同地点、分句顺序、附带专业、未来计划、他人主体、假设、疑问、撤回、暂时限定与冲突目的地。相关1150测试通过、2944未选、4既有警告；.tmp/mechanism-r130-memory.xml。Ruff和定向diff通过。

## 真实强模型测试

服务器隔离快照 /home/boot/lhm/multipersonal-runtime/evaluations/mechanism-r130-source；输出 r130-clause-transfer。4场景12轮HTTP200，12回答、12writer、12行为策略、8证据筛选，共44次DeepSeek调用；语义复核未触发，0意外本地生成，无LoRA、无RAG，冷读保留原文召回。不是生产队列/鉴权/负载验收。

1. 绍兴→绵阳、惠州→遵义、南通→芜湖：均实际写成 SUPERSEDE，新记录active，旧记录superseded并有结束时间。不是仅回答成功。
2. 妹妹搬包头、本人计划明年潍坊：本人现居茂名未被替代，但新暴露妹妹地点被写到 user_location，不能算负例完全通过。
3. 新现居及专业在准备上下文时仍被降为 observation，语义判断不认作当前，回答出现“不确定现在是否还有变化”。只声明三条版本更新成功，不声明整个问答链路成功。

## 下一步已定位的问题

- temporal_projection.project_temporal_record 要求提取器的固定content与writer自然表述逐字相等；还要求提取证据覆盖整个quote。即使已完成更新，自然表达也被投影成“时效未核实，不代表当前状态”。这影响长期记忆→动态上下文→最终回答。
- inference.memory_response._value 同样依赖固定显示模板；需要以有来源的字段语义而非显示文本确认值，并保留条件、历史、冲突及主体边界，不能全局把观察记录当当前事实。
- memory_subject._DIRECT_SUBJECT 的谓词列表缺搬家，故“我妹妹已经搬到包头了”未判定其他主体。writer确实提出了错误location种类，准入没有挡住；模型与机制均有责任。原文可保留，但不能提升为用户自身字段。
- 删除操作识别/执行回执尚未修改；API的complete_turn在回答后运行且忽略执行结果，必须继续核对真实操作路径。
- RAG、动态上下文及跨模块后端均仍未验收，不能用本轮记忆回归替代。

本地原始证据：.tmp/mechanism-r130-transfer-{manifest.json,traces.jsonl}。fixture：backend/evaluation/fixtures/clause_update_transfer_20260927.json。远端只修改授权隔离目录；生产未变。评测已结束，SSH transport已退出。目标保持active。
