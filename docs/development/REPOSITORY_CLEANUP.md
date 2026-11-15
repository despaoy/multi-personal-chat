# 仓库清理清单

截至2026-11-15，已按用户确认范围分六批物理删除769个文件，并从Git索引移除；尚未提交或推送。原软件功能保留，未改动服务器文件或进程。本文件只记录当前状态，替代早期“仅停止跟踪、本地保留”的阶段性说明。

## 已执行范围

| 批次 | 删除范围 | 文件数 |
|---|---|---:|
| 01 | docs/reports历史报告420份、docs/maintainer材料3份 | 423 |
| 02 | scripts/archive、archive/p6_rag_pipeline、deploy/archive/legacy | 50 |
| 03 | docs/testing顶层带日期、未发现代码和配置引用的历史报告 | 238 |
| 04 | 5份生成HTML审阅页面、3个专用HTML生成脚本 | 8 |
| 05 | 5个退役候选生成脚本、5份历史服务器状态快照 | 10 |
| 06 | V5的30份重构过程材料、5个专用工具、5份专属测试 | 40 |
| 合计 | 均已核对工作区路径、普通文件类型和原SHA256后删除 | 769 |

逐文件清单保存在本地忽略目录local-notes/cleanup-2026-11-15/batch-01.json至batch-06.json。历史版本可从提交edf48c54aa71db64e2be3307cf54fd86c8d40379恢复；没有重写Git历史，因此不代表历史仓库体积已经缩小。

V5仅保留candidates.jsonl中的12条待审核候选和简短README。12是实际记录数，原来的13是改写任务数量；这些候选未升级为正式训练数据。候选JSONL与V4正式训练、验证数据的哈希经核对未变。

## 随清理完成的修正

- 已删除文档的有效历史链接指向清理前Git提交；脚本索引移除退役入口并补登记现有LHM工具。
- 仓库完整性检查移除已退役归档的专用检查，继续保留冻结训练数据、API路由、导航及脚本索引检查。链接与脚本索引核对Git可用文件集合，避免本地忽略副本掩盖缺失。
- 发布检查允许.env.character-chat.example配置样例。
- test_kisaki_v4_pipeline.py移除25项只服务于退役生成器的测试，保留共享hard gate、judge与客户端测试。
- probe_lhm_runtime.connect移除个人桌面路径硬编码：优先读取显式LHM_SSH_CONFIG，否则沿用remote_config的LAB_*连接配置。原调用接口保留，实际SSH只读连接验证通过。
- scripts/local-verify.ps1为Python及其子进程设置UTF-8，解决Windows默认GBK导致的测试收集解码错误。

## 保留边界与剩余候选

用户要求保留原软件功能，以及F训练/实验管理、G QQ/NoneBot/AstrBot、H正式训练数据/Gold/批准证据。

| 对象 | 审查结果与处理 |
|---|---|
| experiments/v4 | 510文件、约14.49 MiB，混合正式数据、增补候选和批准来源；不能整目录删除 |
| dpo_expansion_100_20260926 | 9文件、约7.11 MiB；promote工具仍读取candidate_review和generations，保留批准来源链 |
| v4_dpo100_20260926 | 9文件、约4.15 MiB；清单状态approved_ready_for_training，948条训练记录及冻结验证集引用，保留 |
| experiments/archive | 32文件、约3.41 MiB；相关旧生成入口已删除，但训练数据与证据引用尚需逐项核查，本轮不删除 |
| review_packets | 多份批准记录仍由冻结门禁与测试读取，保留实际输入和批准证据 |
| DPO100执行工具 | preflight、train_stage、queue、evaluate、remote构成已批准数据的复现入口，保留 |
| LHM验证工具 | SSH连接、remote_python、真实embedding验证存在交叉调用，保留 |
| backend/tests | Python测试未发现字节完全相同的重复组；尚未证明语义重复，不因数量多而删除 |
| 重复数据文件 | 部分候选、原始记录和批准数据内容相同，但承担不同流程角色；不自动去重 |

继续保留数据库迁移、锁文件、配置样例、核心测试、真实集成验收入口、正式需求、当前架构与部署恢复文档，以及模型tokenizer和许可证。

## 验证结果与限制

- 仓库完整性、发布文件检查通过，发布检查报告errors为空；全项目F821/F822/F811检查在V5清理后通过。
- 第05批相关共享流程、研究与发布测试106项通过；第06批研究、canonical与Gold相关测试35项通过。
- 本轮TypeScript检查通过，直接运行项目已安装的typescript/bin/tsc；pnpm入口因Corepack尝试下载包管理器时DNS失败而未能运行，不能将其记为pnpm成功。
- PowerShell验证脚本语法检查、Git工作区差异空白检查通过。
- UTF-8环境下全量测试收集得到11247项测试，但仍有8项收集错误：本地环境缺少scikit-learn、NoneBot及transformers。它们已在backend/requirements.txt声明；未删除相关功能或测试来消除错误。此次收集不是全量测试通过。
- 当前未发现测试收集错误指向已删除的V5或旧生成脚本，但这不证明所有运行路径完整。
- 全新检出构建、完整应用启动、完整后端测试与生产构建验收尚未完成。本轮未新增删除文件，未推送GitHub。

## 从入口追踪代码：第一批

检查前端layout/page、AuthProvider/AuthGuard、API客户端及代理入口；检查后端run.py → app.main/create_app → runtime/lifespan及鉴权、统计链路。前端通过TypeScript模块解析核对静态导入、再导出和字面量动态导入，并搜索调用名称；api-rag-contract.d.ts是模块类型增强，不能按没有显式导入删除。后端app/api/services/repositories未发现F401/F841问题；顶层符号名称扫描也未发现可直接删除的无引用函数或类，扫描结果不等于所有动态调用均已逐行审查。

本批改动：
- src/lib/api.ts移除8个无调用包装方法：listModelConfigs、getTrainingTask、listStyles、getKnowledgeDocument、updateKnowledgeBase、getRouterConfig、updateRouterConfig、getRouterLogs；连同api-contracts.ts移除11个独占接口类型。后端对应接口全部保留。
- 将JSON请求、健康检查、数据集下载的fetch配置与401处理集中到requestResponse。JSON错误结构、health失败信息、下载原始Response及未消费的响应体保持原行为。
- 两份产品代码净减少162行；增加2项有行为断言的下载/认证回归测试。

验证：TypeScript（含noUnusedLocals/noUnusedParameters）、全前端ESLint、47项前端测试、87项后端启动/就绪/统计测试、仓库完整性与git diff --check通过。前端代理测试在沙箱内因本机连接受限失败，允许本机测试连接后47项全部通过；未改变代理实现来迁就环境。

范围限制：本批不是整个仓库逐文件审查完成；训练、机器人、检索和聊天服务的内部实现仍需继续按具体入口追踪。没有运行生产构建，没有提交或推送。

## 持续精简目标与工程保护边界

用户于2026-11-15要求设置持续目标：逐步精简未使用代码，同时保持高并发、高可靠、高安全能力。目标已建立，尚未完成全仓库审查。

每批删除前核对入口、直接引用、动态注册/框架回调和运行配置；不能将“仅测试调用”直接判为无用。保护排队与背压、限流与容量上限、幂等与会话串行化、超时取消、重试熔断、资源释放、鉴权与租户隔离、事务/修订冲突、监控和错误报告。

本轮检查聊天任务与SSE入口，以及应用、服务、仓储、角色、推理、基础设施、数据库、缓存和训练目录中的无引用候选。保留训练框架on_train_end回调、监控get_counter和缓存cleanup_expired；后两者仍服务于监控并发及缓存回收测试。聊天任务的去重、容量限制、shield断连保护、shutdown释放和SSE所有权检查均未更改。删除useEvaluation中未被消费的GoldPrompt类型别名及其专用导入，保留实际GoldPromptRecord数据契约。

验证：78项现有队列、并发、鉴权背压、限流容量、缓存与监控测试通过；5项评估Hook回归和TypeScript检查通过。这是功能回归基线，不等同于生产压测或全量安全审计。后续仍需逐条深入训练、机器人、检索、聊天内部链路；当前不将静态名称扫描称为全仓库审查完成。

## 训练与机器人入口：参数透传精简

沿NoneBot消息/工具注册、AstrBot插件入口及SimpleLoRATrainer任务分发检查。工具注册表处理函数、训练回调均有动态入口；未删除。检查到的相同函数体主要是协议声明和独立评估探针，不能据此直接合并运行组件。

SimpleLoRATrainer._build_config原先对额外字段先按名单透传，再通用透传；合并为一次通用透传，保留显式转换字段、旧别名优先级、mixed_precision排除、target_modules解析和8位优化器映射。未改变准入、取消、保存、关闭或GPU保护流程。产品代码净减少14行。

验证：对比HEAD版本和工作区版本，256种参数组合的配置构造参数完全一致（替换的是配置构造器以检查边界参数，未加载模型，不代表训练运行验证）。现有生命周期及导出并发测试23项通过；1项训练回调测试因本地缺少transformers失败，未删除或跳过该测试。相关Ruff和仓库完整性通过；训练完整验收仍待依赖齐备后进行。未提交或推送。

## 检索入口：注册表无调用方法

追踪api.ask到grounded_answer和multiscale_rag；核查知识库模块无调用候选。删除KnowledgeDomainConfig.resolve_index_root和DomainRegistry.domain_ids，当前工作区代码、脚本、测试与文档未发现调用；实际运行索引路径由MultiScaleRagRuntime管理，域选择使用配置列表。保留配置数据字段和既有注册接口。game_rag审阅读写工具仍服务于批准材料验证，保留。intent_trainer中未使用绑定的SentenceTransformer导入实际上用于提前检测依赖，未直接删除，以免改变失败路径。

验证：域绑定、实体/关系范围、查询别名、索引失败处理、导入边界、证据装配和多尺度运行测试共169项通过，2项因本地索引embedding_encoding与编码器契约不符失败。独立进程加载HEAD原注册表重跑这2项，复现同一失败，证明不是本批删除引入；未修改索引清单或放宽校验。Ruff、仓库完整性和差异空白检查通过。索引构建验收仍未完成，不能声明检索运行整体验收通过。目标保持进行中，未提交或推送。

## 完整测试收集与残留辅助方法

继续沿依赖关系删除DomainRegistry.list_domains：其唯一产品调用方domain_ids已退役，test_semantic_history_search中残留的注册表替身从未被被测链路读取，连同专属类型导入移除，原断言保留。更正此前概括：实际MultiScaleRagRuntime通过registry.require获取明确域，QueryAnalyzer接收配置列表，不是调用注册表的list_domains。

补齐项目.venv依赖，采用仓库CI的requirements-preference.txt版本及声明范围内的NoneBot/scikit-learn/sentence-transformers，保持torch==2.8.0+cpu；fsspec调整为2025.9.0、huggingface-hub调整为0.36.2以满足兼容要求。没有改系统或服务器依赖，下载缓存位于忽略的local-notes/uv-cache。uv pip check确认111个已安装包兼容。

完整后端测试收集成功：11328项、0项收集错误，日志为本地collection-dependencies.txt；这不是11328项执行通过。训练生命周期/模式/恢复/数据库绑定、NoneBot与历史检索执行84通过、2跳过、1失败；唯一失败为Windows合成JSONL默认写入CRLF而测试计算原始字节哈希，生产校验按文本规范化造成差异。测试样本显式使用UTF-8和LF，生产哈希校验未变；重跑恢复测试文件6项全部通过。此前缺transformers的训练回调测试本轮已通过。真实GPU训练、外部服务和本地索引编码契约问题仍待最终验证，目标未完成。

## 隔离构建与完整后端执行

从Git发布文件集合（当前索引+未忽略新增文件，过滤已删除文件）复制2827份源文件至本地忽略目录local-notes/verify-source-20261115；node_modules仅通过目录联接复用现有依赖。副本不含主工作区.env及开发构建缓存，不是重新安装依赖的全新检出。Next 16.4.1生产构建、TypeScript检查、25个静态页面生成及代理路由构建通过。副本Python源码compileall通过。

本地冒烟脚本仍要求mock生成成功，实际生产契约明确拒绝mock推理。修正脚本为验证健康200、明确mock不可用503、无伪造历史记录；移除无效的成功回复字段断言和允许丢失历史的宽松分支。使用独立临时SQLite、显式合成密钥和独立备份目录，强制SQLite以隔离当前数据库配置；本地诊断文件保留至进程结束后处理，避免Windows线程连接持有文件导致退出清理报错。此脚本不证明真实模型推理成功。隔离副本执行通过，验证脚本标签同步更正。

完整后端执行（沿用CI排除fault_injection_test.py的范围）：11187通过、44跳过、29失败、68错误，耗时368秒。日志local-notes/cleanup-2026-11-15/full-backend-tests.txt。部分错误明确来自STAGE34_DELIVERY_PG_URL等PostgreSQL测试连接变量未配置；失败包括Windows符号链接权限、文件替换语义、本地索引编码契约，以及需要复核的鉴权、记忆回收和超时断言。尚未完成逐项基线归因，不能称整体验收通过，也不能一概视为环境问题。全项目F821/F822/F811、仓库完整性和git diff --check通过。目标保持进行中；没有提交推送。

## 完整回归失败复核：可靠性与账号边界

备份恢复失败定位为测试连接未关闭：sqlite3连接上下文只提交/回滚，不关闭连接；改用closing并保留事务上下文。生产原子替换、校验及安全副本逻辑未变。生成超时测试仍以30ms验证锁等待拒绝、无模型调用及信号量完整归还；成功恢复阶段还原正常超时预算，避免用30ms验证包含异步模型工作的正常请求。两个测试文件20项通过。

架构默认管理员测试的例外清单早于账号自有角色库/聊天功能。显式列出22个已有自有写接口，逐个检查实际account依赖和嵌套get_current_user函数身份；不按路径前缀放行，新接口仍需显式审查，其他写接口仍默认管理员。结合跨账号拒绝和角色/聊天回归共49项通过。生产鉴权、所有权校验未变。

两项记忆来源/跨来源删除失败独立重跑通过；尚未证明全量时序稳定，不放宽删除边界或撤销时间校验。相关Ruff和差异检查通过。当前全量结果仍是此前11187通过等统计，不能将局部重跑推算成新的全量通过数。目标继续，尚未提交推送。

## 独立PostgreSQL验收与来源测试可移植性

只读核查已授权服务器lhm目录后，移除run_lhm_postgres_tests.py对原服务postmaster.pid和/proc进程环境的依赖。postgres二进制的系统库可解析，但initdb还需要lhm内pgsql工具包的libpq；现显式指定该工具库目录，保留路径限制、随机本机端口、独立数据目录及finally停止流程。第一次在缺少libpq配置时于initdb前失败，补齐显式库路径后在/home/boot/lhm/multipersonal-deploy/pg-foundation-x2wljura完成27步迁移、13项真实PostgreSQL聊天/资料测试，pg_ctl确认server stopped。没有启动原数据库、改动其数据或中断训练任务。临时验证文件保留于lhm内。

知识库错误协议测试按操作系统检查真实目录打开异常（Windows PermissionError，POSIX IsADirectoryError），仍严格验证目标文件名、HTTP500、错误协议、敏感路径不泄漏及恢复后结果一致。来源快照合成文件显式写入LF，使BODY字节哈希的假设跨平台一致；生产读取与哈希校验未变。两个测试文件61项通过，相关Ruff与差异检查通过。

当前13项真实PostgreSQL验收未覆盖全量中68项缺少专用PG连接配置的记忆相关测试，不能据此宣称这些错误已消除；仍需独立数据库和相应测试配置。目标继续，未提交推送。

## 最终验证与同步准备（2026-11-15）

本轮定向后端回归169项通过，覆盖训练生命周期与恢复、账号权限边界、备份恢复、生成并发、知识库所有权、来源证据、历史检索、推理队列与限流容量。独立PostgreSQL完成27步迁移，聊天、来源及记忆边界共246项通过；日志确认server stopped。测试专用数据库与文件均位于服务器lhm目录。跨来源删除测试显式使用删除之后的时间，避免系统时钟粒度造成同刻事件；生产删除围栏未放宽。

前端47项测试、TypeScript及未使用符号检查通过；此前隔离发布副本的生产构建与本地冒烟通过，随后未修改前端源码。仓库完整性、发布卫生检查、Python F821/F822/F811和Git差异空白检查通过。发布文件集合2827份，未包含本地测试日志和凭据。

此次提交包含769份已确认退役文件的删除及相关入口、链接、测试修正；正式训练数据、原功能入口、并发和安全保护保留。next-env.d.ts的本地开发自动生成变化不纳入提交。详细测试日志和删除清单留在忽略的local-notes目录。

验证限制：没有在修正后重新执行整个后端测试集，不能宣称全量通过。此前全量结果仍以本记录原始统计为准；Windows符号链接权限、bash可用性、打开文件的原子替换差异、个别子进程事件循环及本地RAG索引编码契约问题仍待相应环境复验。真实GPU训练和真实模型推理不在本轮最终验收范围。
