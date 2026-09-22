# 冻结状态复核版本

`semantic_state_v1_20260919.patch` 保存本轮最初状态复核模块相对于 Git HEAD 的补丁，用于在修改词表/完整输入契约之后复查原实验逻辑。

- 基础提交：`759dbe6bb7326d85ad9a72f9e2a85e3f71319052`。
- 模块：`backend/character/semantic_state_estimator.py`。
- 当时本机文件 SHA-256：`7518befb9d9f17489d7c8823608d7b10b6a33947b4d15c1363d037a6ded5cded`；Git 检出换行转换可能改变字节指纹，应核对而非忽略。
- 对应 rules / semantic / semantic_all 首轮状态入口，以及 isolated thinking seed42 v1。

补丁只冻结该模块，不宣称冻结所有项目依赖。其他依赖指纹和推理配方仍以各报告为准。若需重建，请使用隔离检出，不覆盖当前用户工作区、不重置既有修改。

`semantic_state_v2_20260919.patch` 冻结同一基础提交上的 v2 模块差异（感谢词表、完整输入预算、闭集交互定义）。本机文件 SHA-256 为 `f2cf55d05c89e8202f55fc1ef655c8e5793d54495a85e98850073332c206a765`。v1 与 v2 是分别相对基础提交的补丁，不是依次叠加；报告中的源码指纹仍是校验依据。
