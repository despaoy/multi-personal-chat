# 输入拼装与未量化基座对照

2026-09-27，r91。排查r90短上下文仍发生归属错误的原因；不改产品提示词，不发布生产。

## 实际服务输入核对：已完成

新增evaluation/chat_tokenization_probe.py。先读取现有服务OpenAPI确认/tokenize支持messages、chat_template_kwargs、add_generation_prompt与add_special_tokens，然后对r90全部48组唯一输入调用/tokenize。逐个比较实际整数token ID序列，不以相同长度替代内容相同；同时核对r90生成响应usage.prompt_tokens。

- 48/48 token序列与本地模型目录AutoTokenizer.apply_chat_template一致。
- 48/48长度与先前真实生成记录一致。
- 最后消息正确关闭后是assistant起始标签，关闭思考时存在空think前缀。
- 0次新增生成，48次tokenize；不是模型权重身份认证或语义正确性证明。
- /v1/models返回qwen3-8b-instruct-awq，root为runtime/models/Qwen3-8B-Instruct-AWQ，parent=null、max_model_len=8192，未显示LoRA模型。

tokenizer.json与tokenizer_config.json在AWQ目录和现有未量化Qwen3-8B目录完全同哈希；两份模型不是因为磁盘分词器不同而得到不同文本边界。工具版本：torch2.8.0、transformers4.57.6、vllm0.10.2。

产物：.tmp/mechanism-r91-tokenization.jsonl、mechanism-r91-tokenization-manifest.json。相关32项本地测试通过，.tmp/mechanism-r91-local.xml。没有找到需要“修复模板”的证据，不擅自修改服务模板。

## CPU未量化基座对照：已完成

已有本地基座/home/boot/lhm/kisaki-dpo100-20260926/models/Qwen3-8B，仅读取。未下载新权重，不加载LoRA，不占用生产GPU。启动前可用内存约53GB；driver要求至少48GiB可用才运行。原BF16权重在CPU转换float32（不是凭空增加权重精度），限制2线程、低调度优先级、GPU不可见。仅8个core_plain预定义案例，greedy、thinking=false、最大96输出token；用同一消息向当前AWQ服务发greedy请求。model.generate为确定性路径，预训练generation_config中的temperature/top_p/top_k警告表示这些采样选项被忽略，不是额外抽样。

这不是纯量化因果实验：CPU Transformers与GPU vLLM后端不同，未证明两个检查点来自同一修订的逐权重转换。即使未量化版本改善，也不能直接声称所有问题是AWQ导致；更不能把8例当广泛能力验收。

共8次CPU生成+8次AWQ生成，全部完成。8个CPU输入token列表与/tokenize已记录的对应core_plain实际服务token逐项相同。CPU全部以151645结束，生成7–78token，没有碰到96token上限。AWQ输出也逐项人工检查，没有将截断或“没报错”当正确。

| 案例 | CPU未量化 | AWQ服务 |
| --- | --- | --- |
| 用户最终计划 | 正确修车 | 反问还书还是修车，未完成回忆 |
| 角色周日计划 | 自编休息/整理房间，未冒领用户安排 | 自编看书听音乐，未冒领用户安排；与r90采样结果不同 |
| 含糊取消 | 擅定保留花店 | 同样擅定花店 |
| 小说住址 | 区分小说主角与用户，但引导用户继续故事不够贴合 | 错误认定用户住宁波 |
| 朋友陶艺 | 错误归用户，并猜二人同班 | 同样错误归用户 |
| 助手猜错职业 | 没认定建筑师，但继续猜设计师/作家等多个职业，体验不合格 | 同样不知道但继续猜职业 |
| 引用恶意指令 | 正确编织 | 正确编织 |
| 角色先前音乐厅计划 | 正确引用 | 正确引用 |

这进一步否定“只要改成确定性解码就都好”以及“只是量化造成”两种单因解释。不把未知职业后乱猜算作高质量通过，也不把角色自由虚构日常计划当作既有事实已被验证。当前证据支持基础读取能力与拒绝无依据推断存在缺口，同时量化/后端/检查点差异可能影响部分项目；需要独立转移与不同能力模型进一步辨别，不能凭两个改善项直接更换生产部署。

运行进程1550335以REMOTE_EXIT=0退出，随后ps确认已不存在，可用内存恢复约53GB。CPU加载17.82秒；单例23.92–150.24秒，不具备当前CPU配置下的实时对话效率，不能作为默认部署方案。32项相关本地测试通过仅证明诊断代码边界，不证明模型效果。

产物.tmp/mechanism-r91-cpu-{comparisons.jsonl,manifest.json}；前两次partial下载只是中途检查，不作全量验收。远端evaluations/mechanism-r91-cpu-base保留。生产配置、数据库、模型文件均未改动；没有新增后台模型进程。
