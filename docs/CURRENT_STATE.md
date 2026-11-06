# 当前实现与阅读入口

本项目采用模块化单体：FastAPI 承担 HTTP 边界，业务编排、持久化、模型协议和知识检索各自有明确入口。本页只维护当前结构与有效约定，不继续追加阶段过程。更新日期：2026-11-06。

## 先读什么

1. [后端说明](../backend/README.md)：启动方式、目录边界及现行错误与数据约定。
2. [代码知识库](architecture/CODE_WIKI.md)：模块入口和更完整的调用链。
3. [人物上下文与长期记忆](architecture/CHARACTER_CONTEXT_AND_MEMORY.md)与[角色知识检索](architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)：记忆、作品资料和通用知识库的边界。
4. [部署指南](operations/DEPLOYMENT_GUIDE.md)与[服务器目录规范](operations/SERVER_LAYOUT.md)：配置、迁移、运行资产和验收。
5. [实验总览](research/KISAKI_EXPERIMENT_INDEX.md)：数据审批、训练产物及模型质量结论。

## 职责与依赖归属

| 入口 | 当前职责 |
| --- | --- |
| [app/main.py](../backend/app/main.py)、[app/runtime.py](../backend/app/runtime.py) | 创建应用、装配依赖、管理启动和退出。 |
| [app/dependencies.py](../backend/app/dependencies.py)、`api/` | 身份与权限、请求数据库、协议校验及响应转换。 |
| [services/chat_generation.py](../backend/services/chat_generation.py)、[api/generate.py](../backend/api/generate.py) | 会话准入、实际生成接线、用量、存档与完成状态。 |
| [services/character_context.py](../backend/services/character_context.py)、[services/chat_answer.py](../backend/services/chat_answer.py) | 本轮画像、历史、记忆、动态上下文和回答编排。 |
| `repositories/`、`db/` | 持久化接口、SQLite/PostgreSQL 实现及必要事务。 |
| [inference/request_context.py](../backend/inference/request_context.py)、[inference/model_manager.py](../backend/inference/model_manager.py) | 本轮提供商连接、分词器、HTTP 资源和模型调用协议。 |
| [knowledge/index_lifecycle.py](../backend/knowledge/index_lifecycle.py) | 通用知识库索引的数据库归属、修订发布与检索缓存。 |

API 使用所属应用的数据库；模型管理器、上下文服务、训练管理器和通用知识索引的状态不应借用另一个数据库。应用连接池等生命周期资源由运行容器持有。共享的无数据库状态资源，以及经过审核的公共作品资料，按各自明确的用途共享。

当前启动入口是 [backend/run.py](../backend/run.py)。会话锁、幂等和 nonce 仍含进程内状态，要求 `BACKEND_WORKERS=1`。运行路径和部署参数以实际配置及部署说明为准，不从历史测试目录推断。

## 一轮生成怎样流动

外部请求先构造 `MessageRequest`，已接受的用户原文不可再赋值覆盖；身份、历史和追踪元数据按入口职责处理。`ChatGenerationService` 负责准入和会话排队，实际生成实现绑定本轮提供商快照与用量范围。

角色准备加载明确的画像和用户范围，读取历史与候选证据，执行已启用的语义审核，再编译参考资料和动态上下文。`GenerationRequest` 负责完整消息、参数和预算的共享编译，提供商适配器只转换调用协议。配置或依赖失败不会自动切换模型。

上下文服务通过显式数据库获取，按当前 `ProviderContextBudget` 复用该数据库的服务及记忆向量缓存。预算变化后创建新实例，在途请求继续使用已取得的对象；独立工厂在没有请求快照时，同样沿用所传数据库的模型选择。

生成、存档、回写和平台交付分别记录实际状态。平台发送确认后才完成相应的交付记忆任务。缓存或无需模型的业务结果必须保留其来源，不能计为一次新的模型验证。

## 长期记忆、RAG 与上下文

长期记忆保留原始来源、结构化事实和用户/角色归属。否定、时间、条件、例外和引述主体不能在提取、选择、编译或复核中丢失。来源撤回、删除、版本及权限以持久化事实为准，模型回复不成为用户事实的证据。

通用知识库沿 `KnowledgeIndex`、`RAGHelper`、`VectorDatabase` 工作；资料更新后通过修订号和索引发布流程刷新。检索评测也使用当前数据库的同一个索引与检索器，索引未就绪明确失败，不能记作正常零分。合法检索没有命中则保留正常业务结果。

`knowledge/multiscale_rag/` 是经过审核的作品资料入口，`knowledge/grounded_answer/` 负责证据约束回答与引用绑定。公共作品资料、用户知识库和私人记忆不互相提升权限或可信度；改写、摘要或名称对应也不能生成新的来源事实。

自动角色/适配器规则路由已退出，配置接口拒绝重新启用旧规则/意图路由。意图样本与分类器训练管理仍保留，不能仅凭训练管理状态宣称线上自动路由已经启用。

## 错误与资源边界

必需配置、模型、数据库、索引及依赖失败应明确报告，保留原因，不用空结果、默认值、旧缓存或 mock 冒充完成。显式停用的可选能力与真正故障分别处理；mock 只允许明确的开发/测试配置。

已知数据库和系统 I/O 异常由应用处理器统一转换；路由只保留有实际业务含义的错误转换。意图管理的状态、取消及元数据读写不再复制六组宽泛包装，成功结果直接读取约定字段。元数据 I/O 故障沿统一处理器输出公共错误，不复制内部文件路径或存储异常原文。

多步骤操作需要报告已确认和未确认的状态。观察超时、等待取消和底层工作真正结束不同，不能通过重新执行或假定回滚来掩盖不确定结果。退出清理按依赖关系完成，释放失败仍需报告。

部分旧管理接口仍有宽泛捕获和重复接线，应依据真实调用链逐项收敛；本页不把目标规范写成已完成的全量重构。

## 验证与历史

阶段内只运行修改直接影响的检查和必要关联项。测试必须提供完整事实、上下文和配置；修正夹具与修复产品分别记录。完整阶段的 CI 以 [GitHub Actions](https://github.com/despaoy/multi-personal-chat/actions) 对应提交为准，定向通过不等于整体验证通过。

合成编码器、受控传输、真实模型请求、真实训练和长期负载分别说明验证范围。旧响应复用不能计为新的模型验证，数据审批、训练完成和模型质量结论也不能互相替代。

原先累积在本页的完整过程、失败记录、实测范围和阅读说明，保留在 [2026-11-06 历史快照](https://github.com/despaoy/multi-personal-chat/blob/40b6fc87f9b5901ef627b7dec68fa8050e3026b1/docs/CURRENT_STATE.md)。研究审核记录的日期、分数、哈希和批准状态继续按原文件保留，不随当前架构说明重写。
