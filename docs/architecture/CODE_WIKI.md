# 代码知识库

文档核对：2026-09-11。本文说明当前源码的模块职责和调用边界。实验结论见[实验总览](../research/KISAKI_EXPERIMENT_INDEX.md)，运行验证见[发布清单](../RELEASE_CHECKLIST.md)，工作区与已部署版本的区别见[当前状态](../CURRENT_STATE.md)。

## 总体结构

| 目录 | 职责 |
| --- | --- |
| src | Next.js管理台，页面、组件、状态和API代理 |
| astrbot_plugins/multipersonal_gateway | 标准化平台事件、调用后端、发送回复并确认送达 |
| backend/api | FastAPI接口、鉴权、请求校验和兼容入口 |
| backend/app | 应用工厂、配置、依赖及生命周期 |
| backend/services | 组织模型生成、人物上下文和模型管理 |
| backend/character | 人物画像、情境、策略、输出检查和长期记忆 |
| backend/knowledge | 作品RAG、通用知识库、检索基础组件和证据回答 |
| backend/inference | vLLM客户端、模型、LoRA及提示词边界 |
| backend/training | 对话编码、SFT、偏好训练和任务管理 |
| backend/repositories | 持久化接口与数据库适配 |
| backend/db、backend/alembic | SQLite/PostgreSQL及数据库迁移 |
| backend/infra、cache、middleware | 并发、缓存、安全、熔断和观测 |
| backend/evaluation、experiments、benchmarks、tests | 质量评测、对照实验、性能测试和回归测试 |
| backend/data、gametext | 人物画像、冻结语料、候选数据和原始游戏文本 |
| scripts、backend/scripts | 显式运行的数据、研究与维护工具 |
| deploy | Compose、Nginx、进程管理和启动脚本 |
| archive、scripts/archive | 历史实现，不作为当前生产入口 |
| local-notes | 本地个人笔记和结果快照，不是发布入口 |

文件数量、测试数量与依赖补丁版本容易变化，使用源码清单、锁文件和实际验证输出，不在本文固定计数。

## 启动与应用装配

在backend目录执行python run.py。app/main.py的create_app负责挂载API、中间件与生命周期；backend/main.py保留兼容用途。

- app/config.py、env.py：配置与环境加载。
- app/runtime.py：应用运行时容器。
- app/providers.py、dependencies.py：为接口提供业务服务和仓储。
- app/readiness.py：就绪检查。
- requirements.txt、requirements-dev.txt、pyproject.toml：依赖与开发工具。

后端加载backend/.env；Compose从deploy/.env插值；Next.js本地变量使用根目录.env.local。无论SQLite还是PostgreSQL，BACKEND_WORKERS必须为1；进程内部异步任务并发与Uvicorn多进程不同。

## 消息生成与平台交付

Web请求通过Next.js代理到FastAPI；平台消息通过AstrBot插件到api/integrations.py。services/chat_generation.py组织生成用例，部分核心生成逻辑和兼容入口仍位于api/generate.py，不能声称所有API都已完全去业务化。

人物链路由services/character_context.py准备画像、关系、历史、动态策略和相关记忆，再结合检索证据、生成请求及推理后端产生回复。输出检查、历史记录和状态更新由相应调用入口协调。

AstrBot使用持久化integration_receipts记录生成与发送状态：
1. 对同一次平台事件进行幂等识别。
2. 已生成但未确认发送时复用保存的回复。
3. 插件等待平台发送调用成功，再调用/api/integrations/astrbot/delivery。
4. 首次成功确认后完成角色历史、记忆和关系更新；重复确认不重复更新。

平台调用成功不等于用户已读。发送与确认不构成跨系统事务，确认丢失和重启仍可能造成重复发送或漏一次状态更新。没有原生消息ID时使用事件级ID，不能保证跨进程重建后稳定。

## API文件导航

| 文件（backend/api） | 用途 |
| --- | --- |
| auth.py | 登录和认证 |
| messages.py、user_data.py | 消息历史和用户数据 |
| generate.py、enhanced.py | 聊天生成与兼容编排 |
| ask.py | 证据约束问答 |
| integrations.py | 外部平台消息和发送回执 |
| characters.py | 人物、关系及记忆管理 |
| loras.py、models.py、router.py | 模型、adapter和可选路由管理 |
| knowledge.py、retrieval_eval.py | 知识库与检索评测 |
| training.py、preferences.py | 训练任务和偏好数据 |
| evaluation.py、experiments.py | 评测与实验 |
| stats.py、config.py、claw.py | 统计、配置和工具相关接口 |

准确URL、鉴权要求和请求字段以APIRouter声明、Pydantic模型与运行实例OpenAPI为准。

## 推理与LoRA选择

inference/vllm_client.py通过OpenAI兼容接口调用vLLM；model_manager.py管理模型提供商，generation_request.py与prompt_policy.py组织请求和可信边界。lora_registry.py、lora_utils.py、adapter_checker.py负责登记、名称路径处理与兼容性检查。

api/generate.py中的选择顺序：
- 请求显式指定loraName时按名称选择，不调用自动路由；不存在时报错。
- 未指定时先使用已激活的adapter。
- 只有无显式名称且持久化lora_router_config.enabled启用时，才调用lora_router.py的意图/关键词分支。
- 最终没有adapter时使用default。

固定角色部署应保持自动路由关闭。角色画像仅在适配器具有明确人物映射时进入该人物上下文链路，不能把任意adapter名称都解释成已注册人物。

## 人物上下文与长期记忆

详见[人物上下文与长期记忆](CHARACTER_CONTEXT_AND_MEMORY.md)。

profile_registry加载画像；situation_analyzer分析情境；decision_policy产生策略；context_builder组织上下文；output_guard检查输出。可选语义复核默认关闭，异常或非法输出回退规则状态。

memory_extractor提取规则候选，memory_llm进行后台版本判断和本地校验，memory_service负责回答前检索，repositories/character_memory对接数据库。写入和读取的候选获取方式不同，不应混为“每次取最近10条”。

核对时工作区的写入路径已包含全量可见active记录的相关性匹配，再选最多10条供LLM判断；这是核对前已有的未提交修改。Top-K不能限制此前全量扫描与embedding的成本，尚需规模评测。

## RAG

详见[多粒度角色知识检索](CHARACTER_KNOWLEDGE_RETRIEVAL.md)。

- game_rag：游戏解析、场景边界、元数据与知识卡候选审核。
- multiscale_rag：当前角色作品索引构建和在线检索入口。
- retrieval_core：文档、embedding、查询分析、索引、召回与重排序基础组件。
- grounded_answer：证据包、引用、生成模式和校验。
- rag_helper、vector_db、text_splitter：用户上传的通用知识库。

角色作品知识按story、scene、fact/relation/event和evidence组织，知识卡命中可回填父场景。通用知识库通过knowledge_base_id隔离，不应静默混用角色作品索引。

聊天低置信时生成自然的角色化弃答，仍返回abstained=true、answerMode=abstention和空引用；模型表达生成失败才使用固定兜底。独立证据问答接口保持自己的澄清/拒答策略。归档P6不参与运行时降级。

## 训练与评测

task_manager.py支持前端与标准配置字段别名，并保留小数epochs；数据集目录优先选择train.jsonl/train.json等明确文件，
其他多个候选文件必须显式指定，避免误选元数据。取消信号在加载阶段边界与训练步结束时检查，检测到取消后不进入最终导出；
正在执行的底层模型加载/GPU计算不能立即强停。同名适配器必须等旧worker退出后才可重启，避免写入同一目录。

training/chat_dataset.py统一消息、调用匹配Chat Template并生成assistant监督标签；system/user仍作为输入条件。trainer.py通过PEFT添加LoRA，TRL SFTTrainer执行优化；evaluator.py保存训练期指标；task_manager.py管理任务。speaker_contract.py保存发言者边界；preference_trainer.py与preference_data_schema.py处理偏好训练及其数据契约。

trainer.py中的类默认参数与正式实验配置不同。R1V4 E1文件记录BF16、rank32、alpha64、学习率1e-4、2epochs、batch1、累积8、长度1280，独立validation，NEFTune和packing关闭；calibration使用单独配置。

数据已冻结不等于模型达标。当前E1和recovery无通过质量门禁的adapter，E2-E5暂停。DoRA、RSLoRA、NEFTune或DPO有代码支持，不等于已经完成有效对照并取得提升。

evaluation中的persona、generation、retrieval、safety指标与人工审核互补。检索Hit@K和MRR衡量证据排名，不等于最终回答正确率。experiments负责受控运行，benchmarks是显式性能测试，tests验证代码行为。历史结果应保留原环境、题集与配置。

## 数据库与可靠性

repositories/messages.py、character_memory.py、user_data.py封装业务持久化需求；db/database.py是SQLite实现，pg_database.py是PostgreSQL实现，adapter.py负责选择与兼容。

迁移链尾部为：
- 006_character_memory：人物关系与记忆。
- 007_memory_claims：版本关系、有效时间、证据与作用域。
- 008_integration_receipts：平台生成及发送回执，当前head。

在backend目录执行python -m alembic upgrade head；完整结构结合db/models.py、db/integration_receipts.py与迁移检查。

infra负责有界执行、并发、权限、加密、熔断、备份和观测；cache负责Redis、配置、回复和TTL缓存；middleware/security.py处理请求安全。具体实现与生产验收分开记录，PostgreSQL或Redis可用不自动证明多进程安全。

## 前端

src/app目录对应characters、training、lora、knowledge、history、evaluation、experiments、integrations等管理页。layout.tsx组织外层布局，globals.css定义样式。

src/components保存UI和业务组件；hooks封装数据获取与状态；contexts保存认证与设置。lib/api.ts封装HTTP调用，api-contracts.ts维护共享契约，lib/proxy.ts和app/api的Route Handler实现后端代理。具体入口策略同时查看src/proxy.ts。

前端执行pnpm install --frozen-lockfile、pnpm ts-check、pnpm lint、pnpm build。端口与命令以package.json为准，当前dev/start使用5000。

## 工具与验证入口

脚本逐项用途见[scripts索引](../../scripts/README.md)和[后端脚本索引](../../backend/scripts/README.md)。数据目录见[角色数据说明](../../backend/data/character_dialogues/README.md)。

代码检查、冻结数据门禁、真实模型评测和生产部署是不同验证层级。完整命令及历史记录见[发布清单](../RELEASE_CHECKLIST.md)，扩展模块时遵循[扩展指南](EXTENSIBILITY_GUIDE.md)。
