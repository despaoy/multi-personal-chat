# r139：个人存储查询与外部RAG路由统一

## 修改

存储状态字段语法从inference.memory_response移到character.memory_query共享，原导入保持兼容；覆盖“还/现在/目前”“保存着”等自然形式，仍要求完整闭字段问题，混合/未知问题不匹配。API依赖路由和回答执行使用同一语法，不因未找到记录而查询外部人物资料。

local_context_only仅去掉开头“请记住/记下”管理包装，完整正文仍经过原有个人陈述/外部任务判断，不以包含“记住”一词豁免整段。未知、混合问题继续交给原路由。未修改人物prompt或添加LLM调用。

## 本地测试

149项聚焦通过；广回归2312通过、1跳过、1902未选、4既有警告，147.39秒（.tmp/r139-wide.xml），重叠不累计。Ruff/diff通过。

初版生产路径测试要求“请查一下明天天气”必然进入静态知识库，实际未进入；本轮local_context_only正确返回false，后续检测器却不选择RAG。未将“不被本地依赖判断拦截”等同于“必须查询静态知识库”；保留该未知请求的依赖反例，生产检索测试改用明确资料问题“解释季风形成”，并通过。不宣称天气检索能力已修复或存在。

## 真实API证据

服务器r139隔离快照，全可选LLM接入DeepSeek，无LoRA，实际RAG开启，暖历史，0本地生成。

原3轮场景8新调用（3policy2answer3writer）：

- “请记住专业”：retrieval=not_requested，不再误报abstention。
- 删除专业并问人物关系：claim/source清空，task_composite、RAG=ok，删除确认正确。回答兄妹及恋人，实际证据包括《蓝宝石的存在证明》的恋人对话，不能仅凭新增关系词判为幻觉；但不是全作品时间线核验。
- 保存状态追问：retrieval=not_requested，memory_storage_status，回答模型未调用；正确说明当前长期记忆无专业记录。

姓名/工作地点迁移6轮19新调用（6policy4answer5writer4selection）：个人请求均not_requested，混合人物问题仍ok且3引用。工作地点保存及博物馆召回正确。**姓名状态未通过**：数据库user_name/content=“用户叫夏澄”，证据“请记住，我叫夏澄。”；读取temporal_views.observation=1、field_presence.name=null，导致确定性回复“尚未明确归类”。不是模型写错字段，而是读取整quote匹配将带管理前缀的有效事实降级。较早commentary按表面回复推测通用事实分类，已通过实际记录纠正。下一步应修复读取契约，不改提示词。

证据：.tmp/r139-rag-{manifest.json,traces.jsonl}、.tmp/r139-transfer-{manifest.json,traces.jsonl}。生产未改，远端全部结束，SSH退出。

## 继续范围

管理前缀与完整来源的读取一致性、source-only删除、自然概括准入、暖历史复述仍待修。只确认已测语法范围的RAG路由，不宣布所有个人任务或RAG充分性通过。动态判断效率/消融、失败与重试等完整目标仍active。
