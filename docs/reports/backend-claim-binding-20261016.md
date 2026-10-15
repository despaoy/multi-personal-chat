# 复用已计算的 claim 版本绑定

日期：2026-10-16。基线：`f19eb53fdd81b287f83dd2dc28ad15c14b29c32f`。

每条通过召回筛选的 claim 已调用 read_versions，紧接着来源记录绑定方法又用同一 row 和版本快照重复调用一次，只取相同 claims。全仓确认该方法只有 MemoryService 一处生产调用，测试无直接调用。

现在直接传入已计算的 storage_versions，删除第二次解析、去重及绑定构造。两个生产文件合计 2 行新增、3 行删除；SQL、授权作用域、来源哈希、版本检查和异常传播均不变。没有新增缓存、配置、判断或模型调用。

服务器隔离 SQLite probe 使用真实完整 source、claim、link、语义写入格式的时间来源、作用域 receipt/revision 和 MemoryService 召回。修复前同一 row 的 read_versions 调用 2 次，修复后 1 次；storage/full-source/source-record-pair/source-record-version 四类绑定数量均为 1，分别与真实数据库读取独立推导的预期一致。这里只证明重复构造被消除，不声称整体延迟有某一比例改善。

现有 6 项定向检查通过（0.59 秒）：完整事件链接及证据/版本绑定 2 项、损坏事件链接必须报错 3 项，以及选中 claim 的有效期/关系/证据/来源保留 1 项。没有新增镜像单元测试，也没有重复全量回归。模型和 embedding 构造在 probe 中被禁止。

语法、增量静态、差异及真实数据库后验检查通过后仅重启后端，核实就绪、配置和其他服务，再按 SHA 同步本地与 GitHub。未调用 DeepSeek、未加载权重，不含密钥或运行数据。
