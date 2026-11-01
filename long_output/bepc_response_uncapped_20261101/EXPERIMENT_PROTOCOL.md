# BEPC：取消整题 mismatch 上限的单因素消融

日期：2026-11-01。旧 BEPC、上一轮独立工程优化代码及全部结果保持不变；所有新增代码、缓存与记录都在本目录。

## 固定比较

| 方法 | 定义 |
|---|---|
| strict_shared | 共享 compact logits / terminal skip 的原生 T=0 Strict |
| bepc_cap4 | 上轮 bepc_fast，原对称BEPC，gamma rescue/protection均0.5，每轮2次、整题4次上限 |
| bepc_no_response_cap | 只移除整题4次限制，保留每轮最多2次，其余接受规则相同 |

不使用0.625 rescue，不改变tau=1、kappa=1、eta=1e-4、U8/P4/W32/pool64、关闭残差衰减、图像/Prompt、原生树与KV/位置/attention。当前每棵树为29个Draft候选加root，共30个验证节点；depth3、topk8、num_q2。EOS mismatch保护与最长前缀first-longest规则不变。未选中分支不消耗预算。

取消上限意味着每个合法 mismatch 都由相同 BEPC 分数与每轮2次限制决定，不在累计4次后回退Strict。配置用 `response_mismatch_budget=null`，运行时用正无穷表达不设上限；旧父类构造器的4仅为初始化兼容占位，在prefill/任何接受决策前已替换。不得以大整数冒充取消限制。累计计数仍完整记录。取消上限后不能退休BEPC observer，所以额外特征计算成本必须计入实测。

## 数据与执行

ScienceQA、COCO、TextVQA各20组开发样本、100组正式样本，沿用上轮冻结同一图像、原始长输出Prompt及官方答案/参考。数据已在此前实验使用，属于配对复测，不称独立未见测试集。开发测试图像仍保持原分离。

使用同一Qwen2.5-VL/ViSpec checkpoint、BF16、T=0、seed0、512-token预算与原生after-block停止条件。两张L20按相同image/group分片，每问题内随机化三种方法顺序，各测一遍；各方法每任务/进程热身一次，热身不计分。新进程恢复必须重新热身。预处理/H2D沿用同题复用口径，不计入模型请求时延。

开发共180次、正式共900次生成。开发阶段先检查capped/Strict与旧输出完全一致、uncapped计数/每轮预算和原生模型调用正确，再启动主测试。该阶段不调阈值、不按质量结果筛选。CPU官方评分在全部GPU生成结束后执行，避免与正式GPU计时重叠。

保留所有EOS、截断、解析失败与不理想输出。即使取消上限导致质量下降，也完成冻结的样本集合；遇到技术失败保存证据修复，不能删除失败问题或更换数据。

## 评价

速度：内容token/s、完整同步模型请求时延、decode_seconds及对应吞吐、原生和有效MAT（Draft only与含root两种）、输出长度、allocated/reserved及增量显存、Target/Draft/vision实际调用次数。

预算：每题累计接受mismatch的均值、中位数、P95、最大值、超过4次比例；每轮实际提交mismatch最大值必须<=2；预算绑定次数、完整打分/parent空间构建等计数。速度须包括无上限后持续运行BEPC的开销。

ScienceQA：同一冻结末行`Final answer:`解析器与官方多选Accuracy，保留Correct→Wrong等所有转移。TextVQA：同一解析器与官方VQA-style accuracy，保留连续得分变化。解析失败按0分保留分母，解释段的事实/推理不由答案分数验证。

COCO：整段描述原样输入相同CIDEr-D、SPICE、CHAIR；新SPICE计算使用官方实现与64GiB Java heap，仅扩大资源，不改变指标。只有完整有序样本、标签、输出逐字一致时才能复用同一官方评分；不复用旧时延。CIDEr按完整语料DF，bootstrap固定DF与官方逐图分数。不由指标投票生成Harmful标签，也不把自动指标称为人工事实GT。人工盲评状态仍pending。

## 统计与判断

按image_id分组、5000次配对bootstrap（种子沿用上轮），比较uncapped-cap4、cap4-Strict、uncapped-Strict，报告全部点值/95%CI；每任务每指标三比较做Holm校正。一次计时无法估计运行间波动，不用单个AUROC或新信号选择参数。

沿用上一轮自动非劣效探索门槛：ScienceQA/TextVQA下降不超过2个百分点；COCO SPICE不低于对照95%、CHAIRs增加不超过2个百分点。使用配对单侧95%界，两种对照分别报告。阈值是冻结的研究容忍范围，不能归因于原论文。样本不足时写未建立；零分歧bootstrap退化不能证明未见样本无风险。

取消上限改变的是整个在线策略轨迹。对输出的质量差异只作策略比较，不能将其直接当作单事件反事实Harmful标签。主要决策看实际请求时延、任务质量与截断/错误，MAT或token/s更高不能单独证明更优。

## 追溯与交付

`audit/PRIOR_PRESERVATION.json`保存旧来源SHA；所有新运行源码、配置在生成前冻结。每条结果保持image_id/question_id、输入tensor SHA、配置SHA、原始token、选中路径及模型调用。正式输出完整保存在results/main；报告包含同场三方法与上轮结果的历史对照、全部变化输出和预算分布，不选择性汇报。
