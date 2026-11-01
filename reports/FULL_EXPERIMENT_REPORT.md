# 干净 GitHub ViSpec 五任务解码性能：完整实验报告

实验标识：`vispec-clean-benchmark-20261031`。本完整报告整理日期：2026-11-01（Asia/Shanghai）；实验配置与生成记录保留原始时间。上轮在评分启动时中断，本次仅补完官方评分、统计、报告和审计，未重新生成已完成的答案。

## 结论

本轮已完成五任务各20组开发、100组测试，所有方法重复3次；MME保留完整双问题。主测12,600条请求、开发3,240条、GPU parity 120条，另有400次预热。七方法是用户要求的五个方法家族，加共享工程优化Strict对照及LVSpec开发适配对照。17项必需运行/评分/统计完整性检查全部通过，三次输出差异为0。

当前结果**不支持BEPC在这五个任务上取得优于普通ViSpec/共享Strict的稳定质量—速度优势**。COCO上BEPC的请求时延与共享Strict几乎相同，而CIDEr点估计下降；短答案任务宽松暴露很少，不能用相同Accuracy证明接受规则可靠。Margin4+CCVE在COCO上有小幅时延点估计优势，但质量指标混合，尚不能宣布质量保持或总体更优。

1. **短答案投机收益有限。** GQA和ScienceQA的AR请求分别为225.20ms、237.87ms，比所有投机臂更快；MME和TextVQA的差距较小。MME中BEPC/CCVE无committed mismatch，ScienceQA全部方法都无committed mismatch。这些任务没有充分检验宽松规则。
2. **BEPC独立算法增益未建立。** COCO原生ViSpec为567.04ms，共享Strict为555.65ms，BEPC为555.06ms。BEPC对共享Strict的速度比为1.0011，配对95%CI为[0.9741,1.0270]，Holm p=1.000；不能把相对原生的全部提速归因于BEPC。COCO的CIDEr-D×100由40.35降至34.48，点估计相对下降14.55%；未校正配对区间对应下降，但六比较Holm后p=0.1128，不能声称校正后统计显著。SPICE×100为20.88→20.77，CHAIRs为3%→5%。
3. **Margin4+CCVE存在小幅速度—质量权衡。** COCO时延547.64ms，对共享Strict的描述性速度比1.0146；CIDEr-D×100为39.24，SPICE×100为21.51，CHAIRs为4%。TextVQA分数由84.30%降至83.70%。这些不同方向的指标不构成已验证的质量非劣效；本轮没有预设CCVE对Strict的独立推断检验家族，不追加挑选性显著性结论。
4. **本ViSpec适配中的LVSpec论文比例不稳健。** GQA为63%（Strict为67%），TextVQA为43.50%（Strict为84.30%），COCO的CIDEr-D×100为14.54（Strict为40.35）。TextVQA平均时延还由1029.91ms增至1036.77ms。开发选择已预先对TextVQA和COCO回退Strict；不能将其回退后的质量称为成功的LVSpec宽松策略。此结论只适用于本文公开的ViSpec短树/静态图像适配，不能外推为原视频论文结果无效。

## 请求时延总览

单位ms，均值；完整配对置信区间、吞吐、接受长度和显存在下方逐任务表与CSV。模型请求包含controller构造、视觉编码、prefill、Draft/Target解码及在线特征；PIL/processor/H2D另计，不是完整用户端端到端时延。

| 任务 | AR | 原生ViSpec | 共享Strict | BEPC | Margin4+CCVE | LVSpec论文比例 | LVSpec适配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mme | 402.99 | 412.01 | 399.50 | 400.31 | 399.84 | 409.11 | 409.16 |
| gqa | 225.20 | 276.14 | 263.71 | 264.75 | 263.96 | 265.78 | 265.67 |
| textvqa | 998.78 | 1029.91 | 1017.75 | 1019.11 | 1018.79 | 1036.77 | 1017.71 |
| scienceqa | 237.87 | 290.53 | 278.11 | 278.99 | 278.35 | 281.02 | 281.39 |
| coco | 800.72 | 567.04 | 555.65 | 555.06 | 547.64 | 551.51 | 555.41 |

LVSpec适配在MME/GQA/ScienceQA与论文比例使用相同接受参数；TextVQA/COCO显式关闭宽松接受。共享Strict是用于区分公共工程优化和接受策略收益的必要对照。

## 正确性边界与审稿结论

- **执行完整性通过，严格全项目未见图像主张No-Go。** 本轮开发/测试物理图像完全分离；MME全部有历史运行记录，TextVQA测试3组、COCO测试2组有历史曝光，均保留而未重抽。详见[最终独立审计](../audit/FINAL_DELIVERY_VALIDATION.md)。
- **AR与原生Strict并非已验证逐token等价。** 600道测试题中，TextVQA有1题、COCO有16题的返回token及文本不同，其余任务无差异；共享Strict与原生ViSpec在全部600题完全一致。差异成因尚未建立，不能预先归因于BF16、KV/mRoPE或树注意力。原生/共享等价不能替代AR/ViSpec等价；在定位前，不作完全lossless或与AR逐token一致的声明。详见[AR/ViSpec输出对照及固定顺序样例](../audit/AR_STRICT_OUTPUT_COMPARISON.md)。
- **现有数据支持固定设置的描述性工程比较，不支持跨任务统一优胜、总体非劣效或事实安全。** 100组/任务是Pilot；无显著差异不等于等效。COCO自动指标不能代替人工事实、关系、属性和内容充分性审查。没有新增Judge、人类标签、分类器或融合方法。
- **进入正式优势声明：No-Go。** 需先定位AR/ViSpec输出差异；再以更充分且真正独立的数据验证有功效的质量约束，并确认算法在共享Strict之上带来实际速度收益。当前无需为获得更好结果重调已看过测试集的阈值。

## 数据与附件

- [主结果CSV](BENCHMARK_SUMMARY.csv)：35个任务×方法组合，全部质量指标、请求/解码吞吐、接受长度、输出长度、截断、显存及95%CI。
- [全部预设配对统计CSV](PAIRED_COMPARISONS.csv)：336项比较，差值、CI、中心化bootstrap近似p及Holm校正。
- [性能图PDF](BENCHMARK_PERFORMANCE.pdf)、[质量图PDF](BENCHMARK_QUALITY.pdf)。
- [原始逐请求记录](../results/main/)、[冻结主配置](../configs/frozen_config.json)、[开发集LVSpec选择](../configs/lvspec_selection.json)。
- [统计/输出封印](../audit/STATISTICAL_ANALYSIS.json)、[最终审计JSON](../audit/FINAL_DELIVERY_VALIDATION.json)。

以下为经SHA封印的逐任务详细报告，保留全部方法、指标、置信区间、实现依据和局限。


本报告使用新 clone ViSpec、冻结的任务短答案/单句 Caption 提示和独立开发/测试划分。每任务100组测试、每方法3次完整生成；MME每组保留原始两个问题。全部成绩属于固定子集，不能写成完整官方split成绩。

日期：2026-10-31（Asia/Shanghai）。MME历史曝光与另发现的任务前缀ID别名均保留，冻结输入未重抽样。所有方法仍在同一批图像上配对；历史曝光与本轮dev/test独立是不同问题。因此不能宣称全部测试图为项目未见，统计与质量不以历史曝光状态筛样。以下数量动态读取 audit/DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json，其effective字段覆盖冻结输入中的原freshness标记。

完成模型请求 12600 次；跨重复轨迹差异 0 对。质量固定使用repeat0，未挑选较优重复。所有失败、解析失败、截断和官方未定义指标保留。

原生ViSpec保留原完整leaf logits展开和最后未使用Draft构树；strict_shared对照才享用共享工程优化。BEPC为wR=1、tau1、2/4 mismatch预算；Margin4+CCVE同预算。LVSpec没有2/4 cap，采用每合法branch相对视觉排名与PST，不能将其结果解释为严格预算匹配消融。

实际运行配置：Target为Qwen2.5-VL-7B-Instruct，Draft为对应ViSpec checkpoint；GPU0: NVIDIA L20, torch.bfloat16, PyTorch 2.5.1, CUDA 12.1；GPU1: NVIDIA L20, torch.bfloat16, PyTorch 2.5.1, CUDA 12.1。每个请求只在一张GPU运行，两卡仅分片不同image/group。

温度=0.0，max_new_tokens=128，max_length=2048；原Draft树配置={"total_token": 30, "depth": 3, "top_k": 8, "num_q": 2}。QA短答案与Caption单句提示、图像处理及模型来源见冻结配置和data各split文件；本轮未因任务质量结果修改提示或模型。

AR基线是Target原生串行loop，复用正确的custom KV与mRoPE，不调用Draft。外置runner修正了上游AR辅助脚本new_token未递增、EOS后仍进行无用Target调用的问题；upstream源码没有修改。AR显存仍包含本进程常驻但未使用的Draft模型，因此不是Target-only生产部署的footprint。

| 任务 | 开发无已知历史曝光/组数 | 测试无已知历史曝光/组数 |
|---|---:|---:|
| mme | 0/20 | 0/100 |
| gqa | 20/20 | 100/100 |
| textvqa | 19/20 | 97/100 |
| scienceqa | 20/20 | 100/100 |
| coco | 19/20 | 98/100 |

PASS:600 distinct physical RGB images globally; MME pair shares only its within-group image
NO-GO:MME historically reused; seven additional selected non-MME image aliases documented
上述补充审计未使用历史质量/时延调整样本、答案或标签；新图泛化主张为No-Go，不妨碍本轮内部配对工程比较。

## MME subset

LVSpec开发选择：`lvspec_ratio0.7`；lambda=0.7，disable_relaxation=False。测试集未改变该配置。

开发规则保留论文比例，因此paper与adapted两臂的接受参数相同；它们不能被解释为两个不同算法，其时延差异反映独立请求的测量波动。

| 方法 | MME总分 | 感知 | 认知 | Acc% | 模型请求ms [95%CI] | 相对原生速度 [95%CI] | Draft/轮 | 峰值allocated GiB | 峰值reserved GiB | 无效解析 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 1867.86 | 1460.71 | 407.14 | 77.00 | 402.99 [377.18, 426.62] | 1.02 [1.02, 1.03] | NA | 22.40 | 31.28 | 0 |
| vispec | 1867.86 | 1460.71 | 407.14 | 77.00 | 412.01 [385.93, 436.46] | 1.00 [1.00, 1.00] | 1.74 | 22.40 | 31.28 | 0 |
| strict_shared | 1867.86 | 1460.71 | 407.14 | 77.00 | 399.50 [373.42, 423.96] | 1.03 [1.03, 1.03] | 1.74 | 22.40 | 31.28 | 0 |
| bepc_algorithm | 1867.86 | 1460.71 | 407.14 | 77.00 | 400.31 [374.22, 424.77] | 1.03 [1.03, 1.03] | 1.74 | 22.40 | 31.28 | 0 |
| margin4_ccve | 1867.86 | 1460.71 | 407.14 | 77.00 | 399.84 [373.75, 424.24] | 1.03 [1.03, 1.03] | 1.74 | 22.45 | 31.28 | 0 |
| lvspec_paper | 1867.86 | 1460.71 | 407.14 | 77.00 | 409.11 [380.61, 435.79] | 1.01 [0.99, 1.02] | 2.13 | 22.40 | 31.28 | 0 |
| lvspec_adapted | 1867.86 | 1460.71 | 407.14 | 77.00 | 409.16 [380.65, 435.86] | 1.01 [0.99, 1.02] | 2.13 | 22.40 | 31.28 | 0 |

BEPC相对其余六方法的时延比较：ar 1.007×，Holm p=0.0054；vispec 1.029×，Holm p=0.0012；strict_shared 0.998×，Holm p=0.0012；margin4_ccve 0.999×，Holm p=0.0012；lvspec_paper 1.022×，Holm p=0.0380；lvspec_adapted 1.022×，Holm p=0.0380。

实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：

| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |
|---|---:|---:|---:|
| ar | 0.00 | 0.00 | 2.00 |
| vispec | 0.00 | 0.00 | 2.00 |
| strict_shared | 0.00 | 0.00 | 2.00 |
| bepc_algorithm | 0.00 | 0.00 | 2.00 |
| margin4_ccve | 0.00 | 0.00 | 2.00 |
| lvspec_paper | 10.50 | 0.37 | 2.96 |
| lvspec_adapted | 10.50 | 0.37 | 2.96 |

这些短答案的首token是各方法共同的严格Target root。零或极少committed mismatch时，相同任务质量只能表明当前提示和长度设置下无观测下降；它没有充分暴露宽松门，不能证明门的判别力或事实安全。


AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。

## GQA subset

LVSpec开发选择：`lvspec_ratio0.7`；lambda=0.7，disable_relaxation=False。测试集未改变该配置。

开发规则保留论文比例，因此paper与adapted两臂的接受参数相同；它们不能被解释为两个不同算法，其时延差异反映独立请求的测量波动。

| 方法 | Accuracy% | 模型请求ms [95%CI] | 相对原生速度 [95%CI] | Draft/轮 | 峰值allocated GiB | 峰值reserved GiB | 无效解析 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 67.00 | 225.20 [199.08, 256.78] | 1.23 [1.19, 1.26] | NA | 20.66 | 31.28 | 0 |
| vispec | 67.00 | 276.14 [250.03, 307.18] | 1.00 [1.00, 1.00] | 0.36 | 20.66 | 31.28 | 0 |
| strict_shared | 67.00 | 263.71 [237.63, 294.72] | 1.05 [1.04, 1.05] | 0.36 | 20.66 | 31.28 | 0 |
| bepc_algorithm | 67.00 | 264.75 [238.51, 296.12] | 1.04 [1.04, 1.05] | 0.36 | 20.66 | 31.28 | 0 |
| margin4_ccve | 67.00 | 263.96 [237.87, 294.99] | 1.05 [1.04, 1.05] | 0.37 | 20.71 | 31.28 | 0 |
| lvspec_paper | 63.00 | 265.78 [239.63, 296.61] | 1.04 [1.03, 1.05] | 0.44 | 20.66 | 31.28 | 0 |
| lvspec_adapted | 63.00 | 265.67 [239.58, 296.40] | 1.04 [1.03, 1.05] | 0.44 | 20.66 | 31.28 | 0 |

BEPC相对其余六方法的时延比较：ar 0.851×，Holm p=0.0012；vispec 1.043×，Holm p=0.0012；strict_shared 0.996×，Holm p=0.0012；margin4_ccve 0.997×，Holm p=0.0018；lvspec_paper 1.004×，Holm p=0.5503；lvspec_adapted 1.003×，Holm p=0.5503。

实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：

| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |
|---|---:|---:|---:|
| ar | 0.00 | 0.00 | 1.28 |
| vispec | 0.00 | 0.00 | 1.28 |
| strict_shared | 0.00 | 0.00 | 1.28 |
| bepc_algorithm | 1.00 | 0.01 | 1.29 |
| margin4_ccve | 2.00 | 0.02 | 1.30 |
| lvspec_paper | 11.00 | 0.16 | 1.38 |
| lvspec_adapted | 11.00 | 0.16 | 1.38 |

这些短答案的首token是各方法共同的严格Target root。零或极少committed mismatch时，相同任务质量只能表明当前提示和长度设置下无观测下降；它没有充分暴露宽松门，不能证明门的判别力或事实安全。


AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。

## TextVQA subset

LVSpec开发选择：`vispec`；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能作为成功宽松策略。

| 方法 | Accuracy% | 模型请求ms [95%CI] | 相对原生速度 [95%CI] | Draft/轮 | 峰值allocated GiB | 峰值reserved GiB | 无效解析 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 84.30 | 998.78 [949.88, 1048.83] | 1.03 [1.02, 1.04] | NA | 22.56 | 40.36 | 0 |
| vispec | 84.30 | 1029.91 [983.22, 1076.92] | 1.00 [1.00, 1.00] | 0.79 | 22.56 | 40.36 | 0 |
| strict_shared | 84.30 | 1017.75 [970.86, 1064.78] | 1.01 [1.01, 1.01] | 0.79 | 22.56 | 40.36 | 0 |
| bepc_algorithm | 84.30 | 1019.11 [972.31, 1066.54] | 1.01 [1.01, 1.01] | 0.80 | 22.56 | 40.36 | 0 |
| margin4_ccve | 83.70 | 1018.79 [971.98, 1066.05] | 1.01 [1.01, 1.01] | 0.80 | 22.61 | 40.36 | 0 |
| lvspec_paper | 43.50 | 1036.77 [983.21, 1090.69] | 0.99 [0.96, 1.02] | 1.77 | 22.56 | 40.36 | 0 |
| lvspec_adapted | 84.30 | 1017.71 [970.94, 1064.80] | 1.01 [1.01, 1.01] | 0.79 | 22.56 | 40.36 | 0 |

BEPC相对其余六方法的时延比较：ar 0.980×，Holm p=0.0340；vispec 1.011×，Holm p=0.0012；strict_shared 0.999×，Holm p=0.0360；margin4_ccve 1.000×，Holm p=0.5439；lvspec_paper 1.017×，Holm p=0.5439；lvspec_adapted 0.999×，Holm p=0.0360。

实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：

| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |
|---|---:|---:|---:|
| ar | 0.00 | 0.00 | 3.32 |
| vispec | 0.00 | 0.00 | 3.32 |
| strict_shared | 0.00 | 0.00 | 3.32 |
| bepc_algorithm | 3.00 | 0.04 | 3.31 |
| margin4_ccve | 4.00 | 0.05 | 3.34 |
| lvspec_paper | 54.00 | 1.23 | 6.56 |
| lvspec_adapted | 0.00 | 0.00 | 3.32 |

AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。

## ScienceQA image subset

LVSpec开发选择：`lvspec_ratio0.7`；lambda=0.7，disable_relaxation=False。测试集未改变该配置。

开发规则保留论文比例，因此paper与adapted两臂的接受参数相同；它们不能被解释为两个不同算法，其时延差异反映独立请求的测量波动。

| 方法 | Accuracy% | 模型请求ms [95%CI] | 相对原生速度 [95%CI] | Draft/轮 | 峰值allocated GiB | 峰值reserved GiB | 无效解析 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 86.00 | 237.87 [225.99, 250.33] | 1.22 [1.20, 1.24] | NA | 18.56 | 40.36 | 0 |
| vispec | 86.00 | 290.53 [277.63, 304.24] | 1.00 [1.00, 1.00] | 0.29 | 18.56 | 40.36 | 0 |
| strict_shared | 86.00 | 278.11 [265.23, 291.79] | 1.04 [1.04, 1.05] | 0.29 | 18.56 | 40.36 | 0 |
| bepc_algorithm | 86.00 | 278.99 [266.07, 292.71] | 1.04 [1.04, 1.04] | 0.29 | 18.56 | 40.36 | 0 |
| margin4_ccve | 86.00 | 278.35 [265.46, 292.04] | 1.04 [1.04, 1.05] | 0.29 | 18.60 | 40.36 | 0 |
| lvspec_paper | 86.00 | 281.02 [268.10, 294.89] | 1.03 [1.03, 1.04] | 0.29 | 18.56 | 40.36 | 0 |
| lvspec_adapted | 86.00 | 281.39 [268.47, 295.11] | 1.03 [1.03, 1.04] | 0.29 | 18.56 | 40.36 | 0 |

BEPC相对其余六方法的时延比较：ar 0.853×，Holm p=0.0012；vispec 1.041×，Holm p=0.0012；strict_shared 0.997×，Holm p=0.0012；margin4_ccve 0.998×，Holm p=0.0012；lvspec_paper 1.007×，Holm p=0.0012；lvspec_adapted 1.009×，Holm p=0.0012。

实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：

| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |
|---|---:|---:|---:|
| ar | 0.00 | 0.00 | 1.00 |
| vispec | 0.00 | 0.00 | 1.00 |
| strict_shared | 0.00 | 0.00 | 1.00 |
| bepc_algorithm | 0.00 | 0.00 | 1.00 |
| margin4_ccve | 0.00 | 0.00 | 1.00 |
| lvspec_paper | 0.00 | 0.00 | 1.00 |
| lvspec_adapted | 0.00 | 0.00 | 1.00 |

这些短答案的首token是各方法共同的严格Target root。零或极少committed mismatch时，相同任务质量只能表明当前提示和长度设置下无观测下降；它没有充分暴露宽松门，不能证明门的判别力或事实安全。


AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。

## COCO Caption subset

LVSpec开发选择：`vispec`；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能作为成功宽松策略。

| 方法 | CIDEr-D×100 | SPICE×100 | CHAIRs% | 模型请求ms [95%CI] | 相对原生速度 [95%CI] | Draft/轮 | 峰值allocated GiB | 峰值reserved GiB | 无效解析 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 41.09 | 21.58 | 3.00 | 800.72 [768.33, 833.74] | 0.71 [0.69, 0.73] | NA | 18.31 | 40.36 | 0 |
| vispec | 40.35 | 20.88 | 3.00 | 567.04 [546.71, 587.80] | 1.00 [1.00, 1.00] | 1.96 | 18.31 | 40.36 | 0 |
| strict_shared | 40.35 | 20.88 | 3.00 | 555.65 [535.23, 576.50] | 1.02 [1.02, 1.02] | 1.96 | 18.31 | 40.36 | 0 |
| bepc_algorithm | 34.48 | 20.77 | 5.00 | 555.06 [533.07, 578.38] | 1.02 [0.99, 1.05] | 2.23 | 18.31 | 40.36 | 0 |
| margin4_ccve | 39.24 | 21.51 | 4.00 | 547.64 [528.50, 566.96] | 1.04 [1.02, 1.06] | 2.10 | 18.36 | 40.36 | 0 |
| lvspec_paper | 14.54 | 18.01 | 4.00 | 551.51 [521.28, 590.13] | 1.03 [0.96, 1.08] | 2.98 | 18.31 | 40.36 | 0 |
| lvspec_adapted | 40.35 | 20.88 | 3.00 | 555.41 [535.02, 576.22] | 1.02 [1.02, 1.02] | 1.96 | 18.31 | 40.36 | 0 |

BEPC相对其余六方法的时延比较：ar 1.443×，Holm p=0.0012；vispec 1.022×，Holm p=0.5199；strict_shared 1.001×，Holm p=1.0000；margin4_ccve 0.987×，Holm p=1.0000；lvspec_paper 0.994×，Holm p=1.0000；lvspec_adapted 1.001×，Holm p=1.0000。

实际宽松暴露与输出长度（响应比例以3次生成的全部请求为分母；计数仅指最终提交的mismatch）：

| 方法 | 含committed mismatch的响应% | committed mismatch/响应 | 可见token/响应 |
|---|---:|---:|---:|
| ar | 0.00 | 0.00 | 23.39 |
| vispec | 0.00 | 0.00 | 23.33 |
| strict_shared | 0.00 | 0.00 | 23.33 |
| bepc_algorithm | 94.00 | 2.74 | 24.52 |
| margin4_ccve | 76.00 | 1.60 | 23.21 |
| lvspec_paper | 100.00 | 8.09 | 29.87 |
| lvspec_adapted | 0.00 | 0.00 | 23.33 |

AR的完整模型请求可因短响应而快于投机方法；接受Draft长度提高也不保证请求提速。上表保留AR及所有慢于基线的负结果，不按接受长度替代实际计时排序。

## 统计口径与局限

- 每问题先平均3次时延；在相同image/group抽样索引下配对bootstrap5000次。MME按category分层、完整双问题group重采样，MME总分及分域分数按官方category公式重算。
- 表中95%CI是点式区间。六个BEPC比较与两个LV适配比较分别按task/metric作Holm校正；不控制所有任务和所有指标合并后的全局错误率。p为中心化配对bootstrap的双侧Monte Carlo近似，并非精确随机化检验。
- Accuracy分母包括全部问题；不把不同任务Accuracy、MME量纲或COCO指标混合平均。TextVQA使用官方十答案VQA-style分数；MME总分为所含category的100×(acc+acc_plus)之和，14类别上限2800。
- COCO使用当前100图完整参考语料计算一次正式CIDEr-D，bootstrap重采样其固定每图得分；未在每个bootstrap内重新拟合document frequency。开发20与测试100的CIDEr绝对值不跨split比较。SPICE或CHAIRi未定义时保留NA，不能用替代指标或删除样本制造结果。
- [COCO开发指标独立复核](../audit/COCO_DEVELOPMENT_METRIC_CROSSCHECK.md)确认真实image ID、每图5条参考和作者原始CIDEr实现逐图结果一致，无重复缩放或截断错误。开发生成与参考词数不同只是描述性关联；没有据此将低分因果归于提示词、长度或小样本DF。
- token/s使用首EOS前、128-budget内可见文本token数，不含EOS；模型请求分母包含controller/vision/prefill/Draft/Target/在线特征。CSV另存包含EOS的completion吞吐，供与前轮口径对照。PIL/processor/transfer单独计时并跨方法复用，不属于表中模型请求延迟；该列不是完整用户端E2E。decode-only口径和输出长度见CSV。
- Draft/轮为有效接受Draft token总数/有效验证轮数，未加每轮root token；AR为NA。显存列是观测最大值，不附伪精确最大值置信区间；AR包括常驻未使用Draft模型。reserved受CUDA allocator历史影响，不是方法独占需求；allocated/reserved增量另存CSV，不通过每请求empty_cache改变口径。
- 原生ViSpec与strict_shared的差异反映公共工程优化；不能把宽松方法相对原生的全部收益归因于信号或算法。LVSpec-paper-ratio仍是ViSpec短树/静态图像适配，未宣称复现原K10视频/H200设定。
- 每任务100组是Pilot。CI条件于当前固定硬件、每问题3次计时均值和当前参考指标；MME各category仅约7/8组，不能据此证明强非劣效或罕见事实错误安全。20组开发的非下降代理门不证明总体非劣效，测试差异不显著也不等于等效；本轮没有完成有功效保证的非劣效检验。Caption CIDEr/SPICE/CHAIR不能证明人工事实或显著遗漏安全。历史曝光数量和别名例外动态来自补充审计；各任务来源/缺图/排除数量见数据审计。
- 全部方法、全部预设配对和负结果保存在CSV；没有根据测试结果选择风险方向、重新调lambda、选择更快重复、训练分类器或融合信号。

实现与数据依据：[冻结实验协议](../EXPERIMENT_PROTOCOL.md)、[LVSpec移植审计](LVSPEC_IMPLEMENTATION_AUDIT.md)、[官方指标实现审计](METRIC_AUDIT.md)、[数据审计](../audit/DATA_AUDIT.md)。

输出：`BENCHMARK_SUMMARY.csv`、`PAIRED_COMPARISONS.csv`、`BENCHMARK_PERFORMANCE.pdf`、`BENCHMARK_QUALITY.pdf`；配置、输入、评分、源码和原始结果SHA见 `audit/STATISTICAL_ANALYSIS.json` 与各方法评分seal。
