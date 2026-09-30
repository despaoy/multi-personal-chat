# r110：原话局部窗口扩展与边界验证

## 结论

新增按命中来源补取邻近原话的索引读取能力，在服务器真实模型对照中改善了省略式取消、地点更正、取消后恢复。但它不能保证远距离更正覆盖，也没有解决模型主体/事项误判；因此保持实验参数，`source_window_radius=0` 默认关闭，不发布生产。

## 实现及研究依据

本轮直接读取 [LlamaIndex PrevNextNodePostprocessor 官方源码](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/postprocessor/node.py)。其沿 prev/next 关系补取命中节点邻居；另有 AutoPrevNext 使用模型判断方向。本项目只借鉴无需额外模型的邻近扩展，没有引入 AutoPrevNext、图数据库或新的写入摘要。

实现不同于原项目的显式文档节点边：本项目按**同一已授权逻辑范围内的已保存原话接收时间**读取前后1或2条，最多4个命中，各自形成局部窗口。时间并列以身份键作确定性排序，不证明实际发言先后；未保存的聊天不会凭空补回。该窗口不是完整会话，不是已认证的语义依赖，也不是跨页面session隔离。

- `backend/db/memory_source_window.py`：复用 scope/state/time/key 索引，时间范围和同时间身份范围分开seek；不扫描正文，不取全库最新N条。
- SQLite、PostgreSQL及同步适配器、repository暴露一致窗口接口，无新表或迁移。
- `SourceMemoryService` 先取锚点，再扩展、去重、按时间编译；最终仍重读所有来源检查删除权限。
- 单条事实全文去重不会提前丢掉寻找后续修订所需的锚点；独立单条且完整覆盖时仍可去重。
- 原话总预算仍为2400转义后字符。预算不够不会只留旧锚点、截掉长更正；当前整包省略策略仍可能损失信息，不能宣称最优。
- 诊断记录 anchors/windows/radius；明确 `window_semantic_relation=not_inferred`。没有将邻近消息直接应用为事实更新。

## 真实模型测试

Qwen3-8B-Instruct-AWQ，8192上下文，无LoRA、无可选模型复核。隔离SQLite、实际scheduler写入、完整prepare_turn、共享generation及guard；**写入未经过complete_turn的上游门控**，不是HTTP/RAG/生产PG联合验收。

1. 冻结r109的10份writer输出，原9个问题使用半径2窗口：10次新生成（小说问题额外触发一次guard重试）。
2. 新增独立5场景15输入：15次新writer、5次基线生成。
3. 冻结这15份写入输出，相同问题、半径2：5次新生成。

合计15次新writer、20次新generation。没有将冻结输出计为新模型写入。

| 案例 | 基线 | 窗口组 | 判断 |
| --- | --- | --- | --- |
| r109“后一件取消” | 两项都保留 | 保留滁州会议，取消宿迁见客户 | 原话修订补回，回答改善 |
| 地点更正 | 曲阜 | 建德 | 正确采用无查询词重合的后续更正 |
| 取消后恢复 | 重复问题并泛泛确认 | 按原计划参加，时间地点不变 | 三条来源完整，回答改善 |
| 朋友取消、用户不变 | 知道朋友取消但丢掉各自具体事项 | 同样丢掉具体事项 | 两组均已具备全部来源；属于读取问题 |
| 取消键盘购买，不取消看展 | 答“不去” | 仍答“不去” | 窗口补齐三条也未纠正事项/人称理解 |
| 隔三条闲聊的研讨会取消 | 未直接回答 | 错答安排仍在 | 半径2未覆盖后续取消，不能作为默认充分证据 |

其余r109旧问题未解决当前住址过度怀疑、人称冒领和小说混淆。比较完整请求确认：9问中6问首个模型请求逐字一致，仍出现回答差异（包括条件问题这次较好）。这些差异不能归功于窗口机制；seed=0不构成可复现输出保证。

独立5题中新增输入0–116 tokens，检索总耗时约3.37–7.07ms（每库2–5条），不是大规模吞吐证明。

## 验证与产物

- 270相关回归通过，3条既有弃用警告：`.tmp/mechanism-r110-final.xml`。包括旧源在260条后仍可定位、同时间排序、范围隔离、原话完整预算、删除后重读、事实去重与后续来源。
- 真实SQLite/PostgreSQL生命周期、搜索、窗口一致：`.tmp/mechanism-r110-pg-parity.json`。窗口验证260个同时间来源、异角色拒绝、删除水位。临时PG正常停止。
- 合成规模：20000行、每1000行同一时间、4锚点、半径2、50次SQLite读取，中位4.71ms、p95 4.90ms、max 6.89ms：`.tmp/mechanism-r110-window-performance.json`。仅数据库读取，批量构造数据，不代表capture吞吐或人物质量。
- 真实模型：`.tmp/mechanism-r110-{boundary-window,transfer-baseline,transfer-window}-{results.jsonl,manifest.json}`。
- 评测案例：`backend/evaluation/fixtures/source_windows_20260927.json`。

Ruff与diff检查通过。所有远端写入仅在`/home/boot/lhm/`隔离评测目录；未部署、重启或更改生产配置。评测结束，SSH退出。

## 下一步必须补齐

本轮顺链路检查另发现：`complete_turn` 有 fiction/note 上游门控，而此前runner直接调用scheduler；网页类型的 `sourceMessageId` 允许空，仓库中网页调用点尚未发现赋值，后台会将该原话记录为missing_identity。集成平台有自己的来源ID，不应被统一替换。这里是静态证据，尚未完成真实HTTP请求验证，不能断言所有实际网页请求都缺失。

因此下一轮优先补齐 prepare→真实回答→complete_turn 的写入评测，并验证网页来源身份；再做可靠的来源依赖和跨距离修订读取。不要继续只用人工来源ID的scheduler测试声称覆盖用户完整链路。
