# 月社妃游戏原文解析器（P2/P2.1）

- 阶段：P2 逐行状态机解析器（P2.1 修正：便携溯源、禁止静默丢失、覆盖校验、换行契约）
- 日期：2026-08-24
- 代码：`backend/knowledge/game_rag/parser.py`
- 测试：`backend/tests/test_game_rag_parser.py`（48 项：33 合成样例 + 15 全语料回归）
- 契约：`backend/knowledge/game_rag/models.py`（P1/P1.1，未修改）
- 基准：`KISAKI_GAME_RAG_BASELINE_AUDIT.md`（P0 审计）

## 1. 公开接口

```python
parse_script_text(text: str, source_path: str) -> list[ScriptSegment]
parse_script_file(path: Path, *, source_path: str | None = None) -> list[ScriptSegment]
parse_script_directory(root: Path, *, source_prefix: str | None = None) -> list[ScriptSegment]
```

- 只返回内存对象，不写 JSONL、不修改任何文件；文件读取固定 UTF-8；
- **便携溯源（P2.1）**：`source_path` 一律为便携字符串，绝不写入本机绝对路径——
  - `parse_script_file` 未显式传入时默认 `path.name`；
  - `parse_script_directory` 默认使用相对 `root` 的 POSIX 路径；传入 `source_prefix` 时拼接为
    `f"{source_prefix}/{相对路径}"`（如 `source_prefix="gametext/纸上魔法使"` 得到
    `gametext/纸上魔法使/1翡翠的排挤原理.txt`）；
  - 目录内文件按相对 POSIX 路径字典序确定顺序（注意 `"10黑曜石" < "1翡翠"`，`'0' < '翡'`）；
- 同一语料在不同机器/目录下解析，得到完全相同的 `source_path` 与段 ID（有测试固化）。

## 2. 状态机规则

正则只做一件事：行首锚定识别 `[说话人] + 开引号`（`_TAG_LINE_RE`，允许标签前有空白）。
**绝不使用跨全文 DOTALL 正则提取台词**（旧 `SCRIPT_RE` 的吞并问题即源于此）。

| 状态 | 输入 | 动作 |
|---|---|---|
| 无台词开启 | 标签行 | 结束当前叙述块 → 截断未闭合台词（若有）→ 开启台词 |
| 无台词开启 | 非标签非空行 | 追加到叙述缓冲 |
| 无台词开启 | 空白行 | 结束当前叙述块（空行本身不产出） |
| 台词未闭合 | 含对应闭引号的行 | 台词结束：text 截到闭引号（含），span 覆盖起止行；闭引号后内容按 §3.3 判定 |
| 台词未闭合 | 无闭引号的行 | 整行追加为台词续行（叙述行因此进入台词 span，不产出独立 narration） |
| 台词未闭合 | 标签行 | **截断**：前一条按已累积文本输出 dialogue + `unclosed_quote` 告警；新台词正常开启 |
| 文件结束 | 台词仍未闭合 | 同上截断，告警改用 EOF 形式 |

引号配对：`「」→corner`、`『』→double_corner`、`“”→curly`；闭引号取同类型第一个出现（与旧正则口径一致，语料无嵌套同类型引号）。

## 3. 关键口径

### 3.1 换行契约（P2.1 统一）

- 输入的 CRLF、CR、LF **均规范化为 LF**——这是**唯一允许的文本规范化**；
- `text` 保留物理行边界与行内原文（含前导空白），但**不承诺保留源文件的换行编码**；
- 除此之外不修改任何标点、空白、引号或语言内容。

### 3.2 空行

- 空白行（含纯空格行）只用于结束当前叙述块，**不生成空 narration**；
- 空行造成的行号间隔由相邻 SourceSpan 推断；全语料未覆盖行恰好等于审计确认的 32 个空行（有覆盖测试固化，见 §4）。

### 3.3 闭引号后内容（P2.1：禁止静默丢失）

闭引号之后的内容按三级判定（集中实现于 `_after_close_resolution`）：

1. **纯空白**：正常结束，无告警；
2. **仅由同类闭引号重复组成**（如 `……！」」`）：原文排版瑕疵——重复字符**保留进 text**（不丢数据、不改原文），并写入稳定告警 `duplicate_closing_quote: count=<N>`。P2.1 全库扫描发现全语料仅 1 处：`12青金石的幻想图书馆.txt:2540`（`[克丽索贝莉露] 「不要啊啊…！」」`）；
3. **其他非空内容**（如台词后的尾部叙述）：**显式失败**，抛出 `ScriptParseError`，携带 `source_path`、`line_no` 与稳定错误码 `trailing_content_after_quote`。不引入重叠 SourceSpan，也不把尾部内容悄悄并入台词。当前 17 个文件不触发此错误（有全语料测试固化）。

### 3.4 未闭合引号（7 处原文瑕疵）

截断输出的 dialogue：

- `text` 保留实际开引号与已累积文本，**不自动补闭引号、不改写**；
- `SourceSpan` 从开引号行到截断行（下一条标签行前一行；EOF 情况为文件末行）；
- `warnings` 恰好一条，格式集中定义于 `unclosed_quote_warning()`：
  - 遇到下一条说话人标签：`unclosed_quote: next_speaker_line=<行号>`
  - 直到文件结束：`unclosed_quote: next_speaker_line=EOF(file_end_line=<总行数>)`
- 截断后新台词独立解析，**绝不吞并**（全语料守卫：所有 dialogue 文本不得内嵌行首说话人标签）。

### 3.5 SourceSpan

- 行号从 1 开始，物理行计数；跨行台词 span 含闭引号所在行；
- 同文件内段按输出顺序 `line_start` 严格递增、不倒序、不重叠；
- **每个非空物理行恰好被一个段覆盖**；未覆盖的物理行只能是空行（P2.1 覆盖测试逐文件校验）。

### 3.6 稳定 ID（P2.1：去序号、便携化）

```text
id = "seg_" + sha256(f"{source_path}|{segment_type}|{line_start}|{line_end}")[:16]
```

- `source_path` 为便携相对路径（见 §1），**不含段序号、不含本机路径**；
- 不使用随机数或 Python `hash()`（后者进程间不稳定）；
- **为何去掉段序号**：同文件内 SourceSpan 互不重叠，`(segment_type, span)` 已唯一；去掉序号后，ID 成为来源位置的纯函数——前面无关段落的切分变化（如叙述块合并方式改变）不会波及后续段 ID（有专门测试：同一行号台词在两种前置切分下 ID 相同）。

## 4. 全语料回归结果（17 文件）

| 指标 | 结果 | 与 P0 审计对照 |
|---|---|---|
| 段总数 | 28,507（dialogue 17,530 + narration 10,977） | dialogue ✓ |
| 说话人数 | 14 | ✓ |
| 妃台词 | 1,598 | ✓ |
| curly / double_corner 台词 | 8 / 0 | ✓ |
| unclosed_quote 告警 | 恰好 7，位置与审计一致 | ✓ |
| 7 条被吞台词独立解析 | 全部恢复 | ✓ |
| 9 条合法跨行台词 | 全部无告警保留（妃 3 条逐字断言） | ✓ |
| duplicate_closing_quote 告警 | 恰好 1（12青金石:2540，P2.1 新发现） | 新发现 |
| trailing_content_after_quote | 0 处触发 | ✓ |
| dialogue 内嵌行首标签 | 0 条 | ✓ |
| 非空物理行覆盖 | 每行恰好 1 次；未覆盖行 = 32 个空行 | ✓（与审计 32 空行一致） |
| 重复运行 | 两次 `model_dump` 全等 | ✓ |
| 目录 API vs 逐文件解析 | 溯源与 ID 完全一致 | ✓（P2.1） |

7 处截断明细（吞没者 span → 告警）：

| 文件 | 未闭合台词 | 说话人 | 截止行 | 告警 |
|---|---|---|---|---|
| 12青金石的幻想图书馆 | 2485–2487 | 夜子 | 2487 | next_speaker_line=2488 |
| 1翡翠的排挤原理 | 3892–3893 | 岬 | 3893 | next_speaker_line=3894 |
| 2红宝石的天作之合 | 70–71 | 暗子 | 71 | next_speaker_line=72 |
| 4紫水晶的怪异传说 | 326–327 | 彼方 | 327 | next_speaker_line=328 |
| 6芙蓉石的终焉轮回 | 228–229 | 理央 | 229 | next_speaker_line=230 |
| 9白珍珠的泡沫爱慕 | 1641–1644 | 琉璃 | 1644 | next_speaker_line=1645 |
| 日后谈 | 183–185 | 彼方 | 185 | next_speaker_line=186 |

## 5. 工程约定

- 无新增第三方依赖（仅标准库 hashlib/re/dataclasses + 既有 Pydantic 模型）；
- 单一职责拆分：`_split_lines`（物理行）→ `_ScriptParser.feed/finish`（状态机）→ `_PendingDialogue`（累积态）→ `_add`（构造段）；正则仅 `_TAG_LINE_RE` 一处；
- 告警与错误码集中定义：`unclosed_quote_warning()`、`duplicate_closing_quote_warning()`、`TRAILING_CONTENT_AFTER_QUOTE`、`ScriptParseError`，不在分支里散落硬编码；
- 未修改 `models.py`、P1/P1.1 测试、`scripts/extract_character_dialogues.py`、游戏原文与任何数据文件。

## 6. 已知边界

- 未闭合台词累积期间的空行按原文保留进文本（语料无此情况）；
- 叙述块保留行内前导空白（如 日后谈:184），符合"不清洗、不改写"口径；
- `duplicate_closing_quote` 容忍规则仅覆盖"纯同类闭引号重复"；若未来语料出现 `」 尾巴` 之类混合形态，将按 `trailing_content_after_quote` 显式失败（有意为之，防止容忍范围扩大）；
- 全语料回归测试在 `gametext/` 目录不存在时自动跳过（CI 未携带语料时仍有 33 项合成样例覆盖）。

## 修订记录

### P2.1（2026-08-24，按人工复核意见修正）

1. **便携溯源**：`parse_script_directory` 默认相对 POSIX 路径 + 可选 `source_prefix`；`parse_script_file` 默认 `path.name`；目录按相对 POSIX 路径排序；新增跨目录同 ID 测试。
2. **禁止静默丢失闭引号后内容**：非空尾部内容抛 `ScriptParseError`（错误码 `trailing_content_after_quote`，含 source_path/line_no）。**偏差说明**：复核预期"现有 17 文件不会触发"，实测发现 1 处原文排版瑕疵（12青金石:2540 重复闭引号 `」」`）。为不丢数据、不改原文且保全语料可解析，纯同类闭引号重复按"保留进 text + 告警 `duplicate_closing_quote`"处理，其余非空尾部内容仍硬失败——待人工复核确认此容忍规则。
3. **覆盖测试修复**：`test_every_nonblank_line_covered_exactly_once` 逐文件计算物理行覆盖，断言非空行恰好覆盖一次、未覆盖行恰为审计的 32 个空行；覆盖辅助函数配 3 个合成反例测试。
4. **换行契约统一**：CRLF/CR/LF 一律规范化为 LF（唯一允许的文本规范化），Schema §4.1 同步修正。
5. **稳定 ID 去序号**：`sha256(source_path|segment_type|line_start|line_end)`，不含段序号与本机路径；新增"前置切分变化不影响同位置 ID"测试。
