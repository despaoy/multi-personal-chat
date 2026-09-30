# r142：纯操作不依赖模型上下文与外部 RAG

## 根因与修复

即使已有真实执行回执，API 仍先路由 RAG，生成器仍先构造完整模型输入。这会浪费检索，且在人设/动态/记忆超预算时，使已经完成的删除无法返回确认。

API 根据与输出相同的纯操作回执渲染契约跳过外部检索；生成器在模型编排前处理回执，返回空 messages/generation 的非模型计划。混合任务保留剩余问题全部内容及普通检索/预算检查。不跳过写入授权、不改人物提示词、不新增模型调用。上下文准备阶段仍可能执行内部判断，这一轮不是整个准备链路的性能优化。

## 验证

- 修改前 6 项新测试失败：成功/处理中/失败回执各有上下文溢出、无用检索两个复现。
- 修改后 149 项聚焦通过；扩大回归 2388 passed、1 skipped、1866 deselected、4 依赖警告，152.14 秒，`.tmp/r142-wide.xml`。定向 Ruff 与 diff check 通过。聚焦不与扩大回归累计。
- 隔离服务器 6 轮 HTTP 200，16 次 DeepSeek 请求（6 policy、4 answer、6 writer），无 LoRA、无本地生成。selection/semantic 未触发，不称已测试通过。
- 猫名写入成功；纯删除 claim/source 为空，rag=not_requested、model_invoked=false、messages=[]；追问不重复名字，但仍误路由 character_abstention。
- 专业删除+人物关系：claim/source 为空，RAG=ok，回答同时含确定性删除确认及兄妹/恋人关系，原文确有相应关系对话。这里只确认所取片段支持，不代表整部作品时间线已核验。后续专业保存状态无记录且不调用回答模型。
- `.tmp/r142-manifest.json`、`.tmp/r142-traces.jsonl`。生产未修改、SSH 已关闭。

仍须检查 prepare-after-delete 失败、混合任务后半部分失败时的操作状态可见性、source-only 删除、暖历史、个人问法路由、动态判断完整轮次及消融。持续目标 active。
