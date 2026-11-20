# 角色资料作者更新：前端阶段（2026-11-19）

## 精确依赖与修改前审查

沿用 `character-merge-20261119`，HEAD `f2d87d460de31beab8963baed7ceb34e7c841b9b`，加前两阶段后端累计补丁（SHA256 `4b2c81e9a6e961f9ec4016b41837f8134386a2edbbe36d1ef639b9562f17bc7b`）。开始及交付前与上轮清单的12个文件逐字节核对，保留已有未提交改动，不重新套用第一阶段。没有恢复 Library 副本的必要。

本轮补丁仅含前端、测试及本说明，不包含后端累计改动。须先具备上述后端状态及迁移029，再应用本前端增量。没有提交、推送、部署、触碰 Windows/服务器、启动生产服务或修改长行切块。旧 v1 文件语义不变。

修改前记录保存在交付包 `pre-edit-review.md`：旧页面同时管理角色、设定、作者更新；资料编辑器独立持有草稿，父页不知道资料修改/忙碌，更新设定后按 revision 重挂载可能丢稿。旧文案把资料更新描述为完整替换；旧前端没有 source_merge 类型。分享面板也只有局部 busy。已复核开发基线和后端契约，仓库未找到 AGENTS.md/.agents/skills。

实际安装 Next 16.4.1。依赖按现有锁文件 `pnpm install --frozen-lockfile --ignore-scripts` 恢复，无依赖或锁文件改动。修改前阅读实际 `node_modules/next/dist/docs/01-app/03-api-reference/01-directives/use-client.md`、`01-app/02-guides/testing/vitest.md`、`03-architecture/fast-refresh.md`；客户端事件/状态保留在 Client Component 边界，不依赖重挂载保留本地状态。

## 职责与行为

- `CharacterSourceUpdatePanel` 管理检查、选择、最终预览、确认及错误状态。分类来自后端 `source_merge.actions/conflicts/status/reason`；前端不重新推导三方合并，不拼正文。
- 明确展示保留本地增改删、采用作者增改删和双方一致。每篇冲突复用 `SourcePreview` 查看双方原文，无默认选择；被删一方明确标注“新版不包含此篇”。未知基线、ID碰撞及未来未知冲突类型只读，不能选边绕过。
- 选择完整后请求 POST `/source-update/preview`；最终完整设定和资料都来自服务端。确认方法只发送版本身份、choices、preview_id，不回传正文；旧无冲突客户端方法保留。新网页的无冲突流程也采用空 choices 的最终预览，统一确认保护，依赖本轮指定后端。
- 409/404/422 拒绝确认后清除旧预览、计划和选择，必须重新检查、重新选择。生成预览失败同样不保留旧选择。未知提交结果（网络或其他服务错误）保留准确预览供同一确认重试；保持编辑锁，不允许取消后立即竞争写入。不会把未知结果宣称失败或成功。
- 角色完整设定仍沿用原后端导入行为，并在最终预览展示；逐篇三方比较仅针对独立原作资料，不宣称设定字段也被三方合并。

## 草稿、重挂载与异步审查

资料内容及其载入/成功保存基准提升到父页，按内容比较是否 dirty，不能再用篇数判断。只读载入不算编辑；删除全部仍是未保存修改。只在成功载入或保存后更新基准，失败保留原草稿；成功保存后保留正文便于继续查看，重建作者计划，避免复用旧资料版本预览。撤销资料也刷新计划。

开始更新审核或切换角色时，存在草稿须明确选择丢弃才能继续；取消保留编辑，可先保存再重新检查。另提供单独“丢弃资料编辑，恢复载入版本”按钮。设定保存会产生新角色修订，因此有未保存资料时先拒绝设定保存，提示先保存或明确丢弃资料；避免重挂载清空编辑。已有仅载入未修改的内容可以安全重挂载。

父页同步 ref 闸门与各组件的 in-flight ref 双重防重入；角色保存、文件导入、分享复制/发布、资料载入/保存/撤销和作者审核互斥。审核从检查到取消/提交完成一直持锁，禁用其他编辑、角色切换和分享控件。异步资料保存期间不能竞争导入；相同按钮连点只进入一次。

初始计划查询与资料列表查询有卸载 active 守卫；资料状态轮询在操作/审核期间暂停，轮询不写草稿。资料编辑器重挂载只发生在受保护的角色/修订切换，且草稿实际存储于父页。收到错误不调用 select、不重设草稿。角色列表刷新只改列表，不覆盖当前编辑。

添加 beforeunload 提示和页面链接点击离开保护，但这不是草稿持久化/自动保存功能：浏览器强制退出、崩溃、原生历史后退及所有跨路由机制尚未浏览器验收，不保证恢复。空资料集发布仍受既有后端规则限制，本轮不增加清空资料的新后台语义。

## 验证与复现

Node v24.22.3；pnpm 可用版本11.25.0，项目声明10.34.2，使用既有锁文件未发生升级。测试渲染器和 React 19.2.3 安装到交付目录隔离 test-tools，项目依赖保持不变。

```sh
npm install --prefix /tmp/character-ui-test-tools --no-package-lock --ignore-scripts --no-audit --no-fund react@19.2.3 react-test-renderer@19.2.3
CHARACTER_UI_TEST_MODULES=/tmp/character-ui-test-tools/package.json node --test scripts/tests/character-source-update.test.mjs
CHARACTER_UI_TEST_MODULES=/tmp/character-ui-test-tools/package.json node --test scripts/tests/*.test.mjs
pnpm exec tsc --noEmit --incremental false
pnpm exec eslint src/app/character-library/page.tsx src/components/chat/CharacterSourceUpdatePanel.tsx src/components/chat/CharacterSourceManager.tsx src/components/chat/CharacterSharing.tsx src/lib/api.ts src/lib/api-contracts.ts scripts/tests/character-source-update.test.mjs
NEXT_TELEMETRY_DISABLED=1 pnpm build
```

- 新增16项真实 React hooks/组件交互回归通过；HTTP受控，UI基础按钮/布局受控，不是浏览器DOM或真实全栈验收。涵盖混合选择/删除、无默认、未知基线/ID碰撞/未知类型只读、无冲突、预览/确认过期、精确网络重试、重复确认、只读载入、全部删除草稿、取消/失败保持、保存成功刷新计划、角色保存保护、异步保存互斥、双击、准确确认请求不含正文。
- 全部63项现有及新增前端测试通过（16新增 + 47原有）。react-test-renderer 已弃用警告保留，不隐藏；后续真实浏览器验收不能由此替代。
- TypeScript通过；变更文件及新增测试 ESLint通过。新增测试最初有 module 局部变量触发 Next lint 与未使用变量告警，已修复后重跑全部63项通过。`git diff --check`通过。
- 生产构建完成编译、TypeScript和25页静态生成，最终因 Next 16.4.1 已知漏洞触发 SecurityFatalError，退出1，构建不能记为通过。没有为了通过检查临时升级或绕过安全提示。建议单独安排 Next 安全升级，不属于本轮。
- 后端12个交付文件校验仍与154项通过的上一阶段完全一致；本轮未重复运行后端154项，未改后台语义。
- 未运行浏览器可视/键盘/窄屏、真实 PostgreSQL/OpenSearch/Temporal/模型、生产迁移、跨标签页和原生历史导航验收。过期写入仍依赖已测试的后端事务守卫；下阶段需全栈验证。

## 交付与整合

交付前端独立增量补丁、README、本说明、修改前审查、测试/lint/类型/构建日志、基线清单和应用校验。验证目录先从精确 HEAD 恢复、应用已校验的后端累计补丁，再检查并应用本前端补丁，逐文件比较结果。原后端补丁及交付包保留，不将不可访问的用户 Windows 未提交工作视为已整合。

仅前端阶段完成至上述验证边界，角色库和全栈最终验收仍未完成。没有可调用的 get_usage_limits，不能核实当前周额度，未沿用历史百分比。
