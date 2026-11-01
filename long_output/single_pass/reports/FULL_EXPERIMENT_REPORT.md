# 五任务长输出单次解码性能：完整实验报告

<!-- verified-interpretation-merged -->

## 结果解读与可用性边界

本轮最先需要处理的质量结论是：**GQA官方原表受到明显的大小写格式地板影响**。Strict原分为2/100，其中另54题只因提取答案与gold大小写不同而不匹配；开发Strict及四个LVSpec阈值原分均为0，因此paper .7的开发“不下降”门没有判别力。不能把原2%解释为视觉理解能力，也不能把该门通过称为GQA保质。原始评分、阈值与主表保持冻结；[格式诊断](../audit/GQA_FORMAT_DIAGNOSTIC.md)仅是事后只读诊断。

除这一评测契约问题外，实际解码结果提供了清楚的证据：BEPC相对共享工程Strict提高约2%–3%的内容token/s、约5%–8%的有效Draft MAT，但没有在任何任务建立整题时延的显著优势；Margin4+CCVE的吞吐变化更小且跨任务不一致。论文ratio .7的LVSpec适配取得最高token/s和MAT，同时生成明显更长、更多失败或截断的文本，所有任务整题耗时均增加。不能只按token/s排列“最佳方法”。

## 数据与比较口径

正式测试已完成4,200次模型请求：600题×7方法，每题每方法仅repeat0；MME为100图完整两问，另外四任务各100图。20组/任务开发数据用于先前已冻结的LVSpec选择。没有为本报告补做模型推理、重抽图像或按结果改阈值。[最终交付审计](../audit/FINAL_DELIVERY_VALIDATION.md)和[资源恢复审计](../audit/SPICE_RESOURCE_RECOVERY_VALIDATION.md)均通过。

比较BEPC/CCVE的算法增益，应主要使用strict_shared：它保留Strict决策并共享紧凑logits、跳过终止后无用Draft树等工程优化；native ViSpec用于展示原版实现。native与shared在600题中可见输出完全相同，因此其质量相同；它们与AR并非已证实的逐token等价基线。

内容token/s=可见内容token总数/模型请求总时间，排除EOS；模型请求时间包含视觉编码、prefill、Draft/Target和方法内部开销，处理器/PIL/H2D预处理另记并跨方法复用，因而不是完整用户端到端耗时。有效Draft MAT按首EOS/512预算裁切后计算接受Draft token总数/验证轮数，不含每轮root；原生MAT及含root推进长度见主表。

## BEPC：接受长度有改善，整题时延收益尚未建立

以下相对strict_shared。时延变化负数表示更快；质量差值沿用冻结官方指标，不把不同任务汇总成一个分数。

| 任务 | token/s：Strict→BEPC | 吞吐变化 | 有效Draft MAT：Strict→BEPC | MAT变化 | 请求ms：Strict→BEPC | 请求时延变化 |
|---|---:|---:|---:|---:|---:|---:|
| MME | 57.61→59.14 | +2.65% | 2.064→2.203 | +6.73% | 1814.93→1802.15 | -0.70% |
| GQA | 64.37→65.68 | +2.04% | 2.146→2.283 | +6.36% | 1496.90→1466.85 | -2.01% |
| TextVQA | 41.91→43.24 | +3.17% | 2.007→2.158 | +7.53% | 2353.50→2348.06 | -0.23% |
| ScienceQA | 63.02→64.52 | +2.39% | 2.051→2.156 | +5.13% | 1749.23→1810.93 | +3.53% |
| COCO | 67.56→68.89 | +1.97% | 2.133→2.232 | +4.66% | 2326.94→2350.17 | +1.00% |

预先指定的每任务、每指标BEPC六比较Holm校正后，BEPC相对sharedStrict的内容吞吐和有效MAT差值均为正且p<0.01。该统计证据只对本次固定计时记录与图像抽样成立，不包含重复运行的计时波动；不能写成跨硬件、跨重复已稳定验证。

| 任务 | 吞吐差值token/s [配对95% CI] | Holm p | 请求时延差值ms [配对95% CI] | Holm p |
|---|---:|---:|---:|---:|
| MME | +1.527 [+0.848, +2.228] | 0.0012 | -12.78 [-51.64, +27.57] | 1.0000 |
| GQA | +1.312 [+0.344, +2.280] | 0.0086 | -30.05 [-79.34, +17.64] | 0.3635 |
| TextVQA | +1.328 [+0.549, +2.111] | 0.0048 | -5.44 [-62.01, +49.43] | 1.0000 |
| ScienceQA | +1.503 [+0.639, +2.335] | 0.0016 | +61.70 [-3.50, +129.96] | 0.2607 |
| COCO | +1.332 [+0.600, +2.023] | 0.0024 | +23.24 [-37.32, +83.18] | 1.0000 |

所有整题时延差值的配对区间均跨0。BEPC在MME/GQA/TextVQA观察到约0.2%–2.0%的缩短，在ScienceQA/COCO反而约3.5%/1.0%增加。ScienceQA平均可见token从110.23增至116.84，COCO从157.21增至161.91：更高token/s同时生成更长答案，足以解释为何这两个工程指标的方向不同；这不是在单事件层面证明输出变长的因果机制。

质量方面，MME答案Accuracy从84.5%降至81.5%，子集总分2180.36降至2091.96；ScienceQA85%降至83%；TextVQA71.6%升至72.3%。这些BEPC/shared质量比较均未通过预设Holm显著性门，但“不显著”不表示等价或非劣效，尤其不能忽略MME和ScienceQA的负向点估计。GQA因上述地板问题暂不作能力排名。

COCO的SPICE从0.13413到0.13574，CHAIRs从27%到22%，但CHAIRi从5.585%到5.631%，指标方向存在分歧；所有这些BEPC/shared差值的配对95%区间跨0。不能由CHAIRs下降就宣布视觉事实更安全，也不能将SPICE变化当作人工成对质量标签。

BEPC每响应平均提交3.97–4.00个Strict mismatch，所有响应均有至少一次宽松接受，接近预先设置的每响应4个预算。这个现象提示预算可能成为当前接受增益的约束，但本轮没有预算消融，无法断言放大预算会改善质量或速度。

## Margin4+CCVE：COCO指标更有希望，但尚无稳健速度结论

| 任务 | token/s：Strict→CCVE | 吞吐变化 | 有效Draft MAT：Strict→CCVE | 请求时延变化 | 冻结质量点估计变化 |
|---|---:|---:|---:|---:|---|
| MME | 57.61→58.34 | +1.26% | 2.064→2.162 | +0.53% | 答案84.5%→83.0%；总分2180.36→2170.54 |
| GQA | 64.37→64.28 | -0.13% | 2.146→2.218 | +0.65% | 原分2%→2%，受格式地板影响 |
| TextVQA | 41.91→42.61 | +1.69% | 2.007→2.130 | -0.60% | VQA分71.6%→70.9% |
| ScienceQA | 63.02→62.66 | -0.56% | 2.051→2.100 | +0.19% | Accuracy85%→86% |
| COCO | 67.56→68.60 | +1.54% | 2.133→2.229 | +0.57% | SPICE0.13413→0.13937；CHAIRs27%→21% |

CCVE在MME/TextVQA/COCO的token/s增加约1.3%/1.7%/1.5%，GQA/ScienceQA反而约−0.1%/−0.6%。整题时延仅TextVQA缩短约0.6%，其他任务增加约0.2%–0.7%。因此本次结果不支持“CCVE普遍加快请求”或“比BEPC更快”的统一结论。

COCO同时观察到SPICE上升、CHAIRs下降，以及CHAIRi从5.585%降至4.722%；这是值得后续图像条件盲审的同向自动指标现象。CCVE/shared不是预设的BEPC六比较家族成员，本报告只报告其描述性差值，不用各方法单独CI是否重叠来推断显著性，更不据此宣布总体事实非劣效。MME和TextVQA的负向变化同样保留。

## LVSpec：高吞吐与高MAT并不等于成功加速

该臂是LVSpec视觉相对排名/PST在ViSpec短树上的明确适配，不是原LVSpec架构与硬件的完整复现。paper ratio=.7，合法分支独立排名；不像BEPC/CCVE，它没有相同的每轮2个/每响应4个mismatch预算，不能视为预算匹配消融。

| 任务 | paper token/s | paper有效MAT | 相对Strict整题时延 | 可见token：Strict→paper | paper触顶 | paper答案提取失败 |
|---|---:|---:|---:|---:|---:|---:|
| MME | 87.59 | 3.532 | +41.75% | 104.56→225.34 | 15% | 154/200 |
| GQA | 95.18 | 3.572 | +63.32% | 96.35→232.69 | 19% | 70/100 |
| TextVQA | 68.22 | 3.524 | +25.34% | 98.63→201.23 | 20% | 74/100 |
| ScienceQA | 96.45 | 3.646 | +54.48% | 110.23→260.63 | 27% | 78/100 |
| COCO | 89.90 | 3.304 | +15.82% | 157.21→242.28 | 8% | NA（无答案提取） |

paper臂在五任务内容吞吐增加33%–63%，有效MAT达3.30–3.65；与此同时整题耗时增加16%–63%，输出token增加54%–142%，QA格式失败显著增多。MME原Accuracy18%、TextVQA10.1%、ScienceQA17%，COCO SPICE0.11582低于Strict0.13413。不能忽略这些质量/格式退化，仅用更高MAT支持“更有效宽松验证”。

开发适配的最终结果：MME/TextVQA/ScienceQA/COCO四任务显式disable_relaxation=True，完全退回Strict；因此其质量/MAT等于Strict属于预设fallback，不是LVSpec宽松方法本身保质的证据。GQA保留.7，paper/adapted是相同算法参数的两个命名臂，不能把微小计时差异解释为算法改进；且开发全零分使其门槛验证失效。下一轮应先冻结格式契约再重新执行开发选择，本轮不作事后换阈值。

## 显存、长度与失败模式

按已观测最大allocated峰值，BEPC与sharedStrict相同，CCVE增加约49.0–50.0MiB。相同峰值不证明BEPC特征零成本：视觉编码/prefill等更大的峰值可能遮蔽小的额外临时内存，应结合CSV中的增量与时间记录。任务最大allocated约18.3–22.6GiB；reserved约31.35或40.43GiB，受同进程allocator历史影响，不能按臂将其解释为最低显卡需求。AR内存含常驻未使用Draft，不是仅Target部署占用。

AR、原生/共享Strict、BEPC和CCVE均未触及512-token预算；因此这些方法的长度差异不来自最大长度强制截断。四任务的LVSpec适配Strict fallback也未触顶；GQA适配与paper参数相同，均有19%触顶。AR/Strict/BEPC/CCVE的QA输出平均约78–95词，尽管提示请求100–160词；没有因不满足词数而删样或强制延长。COCO Strict/BEPC/CCVE平均约131/136/134词，paper约184词且8%触顶。将提示要求当作“实际均为相同长度”会误导速度比较。

格式失败与任务错误独立记录：例如ScienceQA CCVE答案提取失败为0，但官方无效选项1题；TextVQA paper提取失败74题而全部官方invalid76题。不能将“解析成功”视为任务正确，也不能只在成功格式子集上计算主分数。

## AR/Strict对齐与COCO指标局限

AR对native ViSpec共有427/600对可见token序列不同，native/shared600对完全一致。实际内容替换均存在，首分叉不在第0 token，因此不能仅用EOS尾部裁剪解释全部差异；387对停止观测字段不同也不等于使用了不同停止规则。现有[只读诊断](../audit/AR_NATIVE_DESCRIPTIVE_DIAGNOSTIC.md)未建立差异来源，不能归因于BF16或某项缓存细节。当前证据支持native→shared保持本轮输出，以及宽松方法相对shared的比较；不支持已证明无损复现AR。

COCO CIDEr-D在长段落上因短参考与长度惩罚落至极小值，Strict约1.95e−43、BEPC2.50e−51、CCVE2.77e−48，paper反而约1.63e−10。不能把paper较大的极小CIDEr当作最好Caption，也不能从BEPC/CCVE CIDEr下降推断新增事实伤害。SPICE/CHAIR同样受不完整短参考、场景图解析与对象词典覆盖限制；它们不评价全部属性、关系、推理或描述充分性。

官方SPICE曾在paper100图长文本评分中因16GiB Java堆上限OOM失败，该无效尝试没有进入结果。仅将同一CPU官方评分进程提高到64GiB并清空失败缓存后恢复成功；输入、参考、JAR与规则不变，GPU解码4200条、计时和配置均未变。它是评分资源故障修复，不是第二轮GPU测试。详情见[资源恢复](../audit/SPICE_RESOURCE_RECOVERY.json)与[独立验证](../audit/SPICE_RESOURCE_RECOVERY_VALIDATION.md)。[缓存警告审计](../audit/SPICE_CACHE_WARNING_AUDIT.md)区分LMDB写缓存警告与真正忽略句子/OOM；前者已有内存解析tuple仍用于评分，不能将二者混同。

## 可支持的结论与下一步门槛

本轮支持：在固定五任务子集、该长输出提示、BF16 Qwen2.5-VL/ViSpec树与一次计时下，BEPC增加有效接受长度并提高记录中的内容吞吐；Margin4+CCVE在COCO有值得人工复核的同向自动指标改善；paper ratio .7在本ViSpec适配中宽松程度过大，token/s/MAT与整题质量和时延明显脱节。

本轮尚不支持：BEPC/CCVE整体比共享Strict更快且保质；人工事实伤害更少；GQA能力排序或.7阈值已保质；AR/native逐token无损；独立全新图像/硬件总体泛化。每任务100组和单次计时的图像bootstrap只能估计样本组成不确定性，不能估计同题重复运行波动。预设检验未显著不等于非劣效；多个不同任务不能合并出平均“综合胜率”。

继续研究的必要顺序是：先修正并在开发前冻结GQA输出格式契约；定位AR/native首分叉；对COCO预先抽样图像进行独立双人盲评并保留uncertain；随后在真正独立的图像上，以预设质量约束与请求时延共同判断是否值得正式比较。本轮不启动这些新实验，也不据现有测试质量调整参数。

原始表、全部配对区间与图见[完整主报告](BENCHMARK_REPORT.md)、[35行CSV](BENCHMARK_SUMMARY.csv)、[配对统计](PAIRED_COMPARISONS.csv)、[吞吐/时延/MAT图](BENCHMARK_PERFORMANCE.pdf)及[质量图](BENCHMARK_QUALITY.pdf)。


报告生成时间：2026-11-01T13:11:27+08:00（Asia/Shanghai）。

## 完成范围与证据

按用户最新要求，正式测试每题/方法仅生成一次。五个数据集各100组测试，MME为完整双问题组，共600题；七个比较臂合计4200条正式请求。开发集九配置×120题=1080条，其中1003条复用已完成的旧repeat0，只新增77条补齐。原repeat1/2保留在旧实验中，不进入本轮主结果，实际已花费的计算不隐去。
所有图像、题目、官方答案、长提示、模型、512-token预算、原树、T=0及既有接受规则不变。统计仅按固定repeat0；置信区间来自5000次图像/组配对重采样，不包含重复计时波动，也不能证明跨运行稳定性。旧核心的120条parity/48组对照作为继承证据，本轮没有额外重跑。

## 核心结果总览

下表给点估计；完整七方法结果、95%CI、原生MAT、有效MAT、显存、解析失败及截断均见下方详细报告。token/s分母包含controller、视觉编码、prefill和全部解码/在线特征；分子不含EOS。MAT仅计接受的Draft，不含Target root；AR不适用。

| 任务 | 方法 | token/s | 有效Draft MAT | 请求ms | 官方任务质量 |
|---|---|---:|---:|---:|---|
| mme | ar | 34.53 | N/A | 3015.46 | MME 2130.36 / Acc 83.00% |
| mme | vispec | 57.37 | 2.06 | 1822.60 | MME 2180.36 / Acc 84.50% |
| mme | strict_shared | 57.61 | 2.06 | 1814.93 | MME 2180.36 / Acc 84.50% |
| mme | bepc_algorithm | 59.14 | 2.20 | 1802.15 | MME 2091.96 / Acc 81.50% |
| mme | margin4_ccve | 58.34 | 2.16 | 1824.57 | MME 2170.54 / Acc 83.00% |
| mme | lvspec_paper | 87.59 | 3.53 | 2572.62 | MME 318.75 / Acc 18.00% |
| mme | lvspec_adapted | 57.65 | 2.06 | 1813.78 | MME 2180.36 / Acc 84.50% |
| gqa | ar | 36.51 | N/A | 2636.00 | 1.00% |
| gqa | vispec | 64.03 | 2.15 | 1504.85 | 2.00% |
| gqa | strict_shared | 64.37 | 2.15 | 1496.90 | 2.00% |
| gqa | bepc_algorithm | 65.68 | 2.28 | 1466.85 | 2.00% |
| gqa | margin4_ccve | 64.28 | 2.22 | 1506.61 | 2.00% |
| gqa | lvspec_paper | 95.18 | 3.57 | 2444.76 | 1.00% |
| gqa | lvspec_adapted | 95.18 | 3.57 | 2444.79 | 1.00% |
| textvqa | ar | 28.29 | N/A | 3520.99 | 70.90% |
| textvqa | vispec | 41.78 | 2.01 | 2360.88 | 71.60% |
| textvqa | strict_shared | 41.91 | 2.01 | 2353.50 | 71.60% |
| textvqa | bepc_algorithm | 43.24 | 2.16 | 2348.06 | 72.30% |
| textvqa | margin4_ccve | 42.61 | 2.13 | 2339.36 | 70.90% |
| textvqa | lvspec_paper | 68.22 | 3.52 | 2949.85 | 10.10% |
| textvqa | lvspec_adapted | 41.91 | 2.01 | 2353.12 | 71.60% |
| scienceqa | ar | 36.56 | N/A | 2995.43 | 86.00% |
| scienceqa | vispec | 62.76 | 2.05 | 1756.50 | 85.00% |
| scienceqa | strict_shared | 63.02 | 2.05 | 1749.23 | 85.00% |
| scienceqa | bepc_algorithm | 64.52 | 2.16 | 1810.93 | 83.00% |
| scienceqa | margin4_ccve | 62.66 | 2.10 | 1752.47 | 86.00% |
| scienceqa | lvspec_paper | 96.45 | 3.65 | 2702.18 | 17.00% |
| scienceqa | lvspec_adapted | 63.06 | 2.05 | 1748.02 | 85.00% |
| coco | ar | 37.52 | N/A | 4197.17 | SPICE 13.59 / CHAIRs 26.00% |
| coco | vispec | 67.41 | 2.13 | 2332.30 | SPICE 13.41 / CHAIRs 27.00% |
| coco | strict_shared | 67.56 | 2.13 | 2326.94 | SPICE 13.41 / CHAIRs 27.00% |
| coco | bepc_algorithm | 68.89 | 2.23 | 2350.17 | SPICE 13.57 / CHAIRs 22.00% |
| coco | margin4_ccve | 68.60 | 2.23 | 2340.14 | SPICE 13.94 / CHAIRs 21.00% |
| coco | lvspec_paper | 89.90 | 3.30 | 2695.08 | SPICE 11.58 / CHAIRs 23.00% |
| coco | lvspec_adapted | 67.61 | 2.13 | 2325.18 | SPICE 13.41 / CHAIRs 27.00% |

## 接受策略收益与工程收益

比较算法时优先参考共享Strict：原生ViSpec额外保留完整leaf logits和最后未使用Draft构树，宽松方法与共享Strict均享有共同工程优化。下面比值只作描述性点估计；统计检验及Holm结论以配对CSV为准，不据MAT或token/s单项宣布总体质量—速度优胜。

| 任务 | BEPC token/s / 共享Strict | CCVE token/s / 共享Strict | BEPC请求速度比 | CCVE请求速度比 |
|---|---:|---:|---:|---:|
| mme | 1.0265 | 1.0126 | 1.0071 | 0.9947 |
| gqa | 1.0204 | 0.9987 | 1.0205 | 0.9936 |
| textvqa | 1.0317 | 1.0169 | 1.0023 | 1.0060 |
| scienceqa | 1.0239 | 0.9944 | 0.9659 | 0.9982 |
| coco | 1.0197 | 1.0154 | 0.9901 | 0.9944 |

本轮AR/native输出对照（仅重读已保存token，未增加模型调用；成因未定位）：

| 任务 | 题数 | AR/native token差异 | native/shared token差异 |
|---|---:|---:|---:|
| mme | 200 | 127 | 0 |
| gqa | 100 | 76 | 0 |
| textvqa | 100 | 69 | 0 |
| scienceqa | 100 | 64 | 0 |
| coco | 100 | 91 | 0 |

## 解释边界与审计

- 本轮为单次测量，速度排名需结合图像抽样CI及输出长度；高token/s可能伴随更长文本、更多错误或更慢请求。没有再执行三次主测。
- QA仅按唯一终止Final answer行评估任务答案，格式/截断失败计0并留在分母；解释文本的事实与推理正确性没有人工Ground Truth。COCO完整长描述用官方SPICE、CHAIR及CIDEr-D；CIDEr长度惩罚可能造成数值极小，不将其单独当事实伤害。
- LVSpec论文比例及开发适配均完整保留。适配若关闭宽松，明确是Strict fallback；不能将其表现称为成功宽松策略。开发集选参不会使用正式测试结果。
- AR显存包含常驻未使用Draft，不是Target-only部署footprint；reserved受allocator历史影响。生成核心、Target/Draft调用、停止、位置/KV和来源映射由独立审计核对。AR/native在前轮存在未定位的输出差异，不能从native/shared一致推出AR/native逐token等价。
- 原始图像开发/测试独立，但MME及已声明的其他历史曝光仍在；全项目未见图像泛化主张No-Go。100组/任务与自动指标不足以证明总体非劣效或人类事实安全。
- 初次单次预检使用base Python/Pillow时出现decoded RGB环境差异，原No-Go记录保留；指定spec Python环境复检通过，原图字节没有更换。正式推理、评测和最终审计使用指定环境。

## 数据与附件

- [完整结果CSV](BENCHMARK_SUMMARY.csv) · [全部预设配对统计](PAIRED_COMPARISONS.csv)
- [吞吐/时延/MAT图](BENCHMARK_PERFORMANCE.pdf) · [官方质量图](BENCHMARK_QUALITY.pdf)
- [独立最终审计](../audit/FINAL_DELIVERY_VALIDATION.md) · [原始运行库存与复用](../audit/ORIGINAL_RUN_INVENTORY.md)
- [单次冻结配置](../configs/frozen_config.json) · [统计/输出SHA封印](../audit/STATISTICAL_ANALYSIS.json)
- [实验协议](../EXPERIMENT_PROTOCOL.md) · [单次质量协议](QUALITY_PROTOCOL.md)

## 逐任务全部结果与统计说明


本轮在相同图像和问题上，将提示改为约100–160词解释/详细图像描述，最大新token预算512。五个任务各20组开发、100组测试，按用户修订每种方法仅完成1次生成；MME每图保留两问。本报告首先比较内容token/s、原生/有效Draft MAT和官方任务质量，同时保留请求时延、长度、显存、格式失败和负结果。

日期：2026-11-01（Asia/Shanghai）。使用前轮fresh GitHub clone和相同模型、数据身份与来源划分；仅长提示和预算改变，开发集重新按同一网格选择LVSpec参数。BEPC/CCVE阈值未重调。所有成绩属于固定子集，不能写成完整官方split或独立全新图像的成绩。

完成测试模型请求 4200 次；不执行repeat1/2。任务质量使用唯一的repeat0；全部失败、截断、无效格式及未定义官方指标均保留。由于每个配置只测一次，本轮没有重复运行稳定性证据；时延和吞吐置信区间只反映图像抽样，不包含重复计时波动。

QA先生成解释，再以唯一的末行Final answer给出最终答案；固定格式提取后调用原官方任务指标。因此QA结果衡量最终答案正确性，无法证明解释正确或推理忠实。COCO整段原样计算SPICE、CIDEr-D、CHAIR，它们是辅助自动质量/对象幻觉指标，不能代替人工事实与信息覆盖评价。

实际运行：Target=Qwen2.5-VL-7B-Instruct，Draft=对应ViSpec checkpoint；GPU0: NVIDIA L20, torch.bfloat16, PyTorch 2.5.1, CUDA 12.1；GPU1: NVIDIA L20, torch.bfloat16, PyTorch 2.5.1, CUDA 12.1。单请求在一张GPU执行，两卡仅分片不同image/group。

温度=0.0，max_new_tokens=512，max_length=2048；Draft树={"total_token": 30, "depth": 3, "top_k": 8, "num_q": 2}。精确文本、输入及源码SHA见冻结配置和协议。

原生ViSpec保留原完整leaf logits展开和最后未使用Draft构树；strict_shared才应用公共工程优化。BEPC固定wR=1、tau1、每轮/每响应2/4 mismatch预算；Margin4+CCVE同预算。LVSpec采用合法branch相对视觉排名和PST，没有相同2/4 cap；这些差别不能作为预算匹配消融解释。

AR使用同一Target custom KV/mRoPE的串行loop，不调用Draft；其显存仍包括常驻但未用的Draft权重，因此不是纯Target生产部署footprint。原短实验已记录AR与native Strict的17题输出差异，本轮保留实际输出，不能假设两个执行图在BF16下逐token等价。

| 任务 | 开发无已知历史曝光/组数 | 测试无已知历史曝光/组数 |
|---|---:|---:|
| mme | 0/20 | 0/100 |
| gqa | 20/20 | 100/100 |
| textvqa | 19/20 | 97/100 |
| scienceqa | 20/20 | 100/100 |
| coco | 19/20 | 98/100 |

这沿用原历史审计相对短实验启动前的状态；现在全部图像已参加短实验，不能再宣称项目未见。本轮开发与测试图像仍分离，历史曝光不用于筛样或改变统计分母。

## MME subset

长输出开发选择：vispec；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能称为成功宽松策略。

| 方法 | 内容token/s [95%CI] | 原生Draft MAT | 有效Draft MAT | 1+有效MAT | 模型请求ms [95%CI] | 最大allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 34.53 [34.17, 34.92] | NA | NA | NA | 3015.46 [2928.81, 3101.74] | 22.40 |
| vispec | 57.37 [56.36, 58.39] | 2.15 | 2.06 | 3.06 | 1822.60 [1771.99, 1871.13] | 22.40 |
| strict_shared | 57.61 [56.60, 58.63] | 2.15 | 2.06 | 3.06 | 1814.93 [1764.31, 1863.66] | 22.40 |
| bepc_algorithm | 59.14 [58.17, 60.15] | 2.29 | 2.20 | 3.20 | 1802.15 [1751.04, 1853.54] | 22.40 |
| margin4_ccve | 58.34 [57.35, 59.35] | 2.24 | 2.16 | 3.16 | 1824.57 [1777.05, 1873.08] | 22.45 |
| lvspec_paper | 87.59 [85.84, 89.20] | 3.57 | 3.53 | 4.53 | 2572.62 [2419.09, 2722.14] | 22.40 |
| lvspec_adapted | 57.65 [56.63, 58.67] | 2.15 | 2.06 | 3.06 | 1813.78 [1763.17, 1862.44] | 22.40 |

| 方法 | MME子集总分 | 感知 | 认知 | 答案Accuracy% |
| --- | ---: | ---: | ---: | ---: |
| ar | 2130.36 | 1508.93 | 621.43 | 83.00 |
| vispec | 2180.36 | 1530.36 | 650.00 | 84.50 |
| strict_shared | 2180.36 | 1530.36 | 650.00 | 84.50 |
| bepc_algorithm | 2091.96 | 1434.82 | 657.14 | 81.50 |
| margin4_ccve | 2170.54 | 1491.96 | 678.57 | 83.00 |
| lvspec_paper | 318.75 | 268.75 | 50.00 | 18.00 |
| lvspec_adapted | 2180.36 | 1530.36 | 650.00 | 84.50 |

| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |
|---|---:|---:|---:|---:|---:|---:|
| ar | 104.13 | 81.85 | 0.00 | 2/200 | 2/0 | 0.00 |
| vispec | 104.56 | 82.39 | 0.00 | 2/200 | 2/0 | 0.00 |
| strict_shared | 104.56 | 82.39 | 0.00 | 2/200 | 2/0 | 0.00 |
| bepc_algorithm | 106.58 | 83.56 | 0.00 | 0/200 | 0/0 | 100.00 |
| margin4_ccve | 106.44 | 83.93 | 0.00 | 3/200 | 3/0 | 89.50 |
| lvspec_paper | 225.34 | 167.72 | 15.00 | 154/200 | 146/8 | 100.00 |
| lvspec_adapted | 104.56 | 82.39 | 0.00 | 2/200 | 2/0 | 0.00 |

BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：ar 1.673×，Holm p=0.0012；vispec 1.011×，Holm p=1.0000；strict_shared 1.007×，Holm p=1.0000；margin4_ccve 1.012×，Holm p=1.0000；lvspec_paper 1.428×，Holm p=0.0012；lvspec_adapted 1.006×，Holm p=1.0000。

token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。

## GQA subset

长输出开发选择：lvspec_ratio0.7；lambda=0.7，disable_relaxation=False。测试集未改变配置。

paper与adapted因预先的paper优先规则使用相同接受参数；它们是独立计时重复臂，不能将时延波动解释为算法差异。

| 方法 | 内容token/s [95%CI] | 原生Draft MAT | 有效Draft MAT | 1+有效MAT | 模型请求ms [95%CI] | 最大allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 36.51 [36.01, 36.93] | NA | NA | NA | 2636.00 [2534.78, 2741.10] | 20.66 |
| vispec | 64.03 [62.39, 65.48] | 2.23 | 2.15 | 3.15 | 1504.85 [1447.85, 1564.23] | 20.66 |
| strict_shared | 64.37 [62.72, 65.84] | 2.23 | 2.15 | 3.15 | 1496.90 [1439.68, 1556.55] | 20.66 |
| bepc_algorithm | 65.68 [64.07, 67.16] | 2.37 | 2.28 | 3.28 | 1466.85 [1413.51, 1522.25] | 20.66 |
| margin4_ccve | 64.28 [62.74, 65.65] | 2.30 | 2.22 | 3.22 | 1506.61 [1451.03, 1564.01] | 20.71 |
| lvspec_paper | 95.18 [92.44, 97.52] | 3.61 | 3.57 | 4.57 | 2444.76 [2179.26, 2721.83] | 20.66 |
| lvspec_adapted | 95.18 [92.43, 97.51] | 3.61 | 3.57 | 4.57 | 2444.79 [2179.20, 2721.55] | 20.66 |

| 方法 | 最终答案Accuracy% |
| --- | ---: |
| ar | 1.00 |
| vispec | 2.00 |
| strict_shared | 2.00 |
| bepc_algorithm | 2.00 |
| margin4_ccve | 2.00 |
| lvspec_paper | 1.00 |
| lvspec_adapted | 1.00 |

| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |
|---|---:|---:|---:|---:|---:|---:|
| ar | 96.24 | 82.83 | 0.00 | 0/100 | 0/0 | 0.00 |
| vispec | 96.35 | 83.17 | 0.00 | 0/100 | 0/0 | 0.00 |
| strict_shared | 96.35 | 83.17 | 0.00 | 0/100 | 0/0 | 0.00 |
| bepc_algorithm | 96.34 | 82.98 | 0.00 | 0/100 | 0/0 | 100.00 |
| margin4_ccve | 96.85 | 83.48 | 0.00 | 0/100 | 0/0 | 99.00 |
| lvspec_paper | 232.69 | 188.78 | 19.00 | 70/100 | 70/0 | 100.00 |
| lvspec_adapted | 232.69 | 188.78 | 19.00 | 70/100 | 70/0 | 100.00 |

BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：ar 1.797×，Holm p=0.0012；vispec 1.026×，Holm p=0.3635；strict_shared 1.020×，Holm p=0.3635；margin4_ccve 1.027×，Holm p=0.3635；lvspec_paper 1.667×，Holm p=0.0012；lvspec_adapted 1.667×，Holm p=0.0012。

token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。

## TextVQA subset

长输出开发选择：vispec；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能称为成功宽松策略。

| 方法 | 内容token/s [95%CI] | 原生Draft MAT | 有效Draft MAT | 1+有效MAT | 模型请求ms [95%CI] | 最大allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 28.29 [27.79, 28.78] | NA | NA | NA | 3520.99 [3402.51, 3643.76] | 22.56 |
| vispec | 41.78 [40.59, 42.95] | 2.08 | 2.01 | 3.01 | 2360.88 [2279.43, 2447.35] | 22.56 |
| strict_shared | 41.91 [40.72, 43.09] | 2.08 | 2.01 | 3.01 | 2353.50 [2271.92, 2439.66] | 22.56 |
| bepc_algorithm | 43.24 [42.06, 44.42] | 2.23 | 2.16 | 3.16 | 2348.06 [2269.16, 2429.47] | 22.56 |
| margin4_ccve | 42.61 [41.44, 43.82] | 2.20 | 2.13 | 3.13 | 2339.36 [2263.38, 2416.97] | 22.61 |
| lvspec_paper | 68.22 [63.81, 72.27] | 3.57 | 3.52 | 4.52 | 2949.85 [2657.90, 3269.73] | 22.56 |
| lvspec_adapted | 41.91 [40.73, 43.09] | 2.08 | 2.01 | 3.01 | 2353.12 [2271.45, 2439.56] | 22.56 |

| 方法 | 最终答案Accuracy% |
| --- | ---: |
| ar | 70.90 |
| vispec | 71.60 |
| strict_shared | 71.60 |
| bepc_algorithm | 72.30 |
| margin4_ccve | 70.90 |
| lvspec_paper | 10.10 |
| lvspec_adapted | 71.60 |

| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |
|---|---:|---:|---:|---:|---:|---:|
| ar | 99.60 | 78.89 | 0.00 | 1/100 | 1/0 | 0.00 |
| vispec | 98.63 | 78.29 | 0.00 | 0/100 | 0/0 | 0.00 |
| strict_shared | 98.63 | 78.29 | 0.00 | 0/100 | 0/0 | 0.00 |
| bepc_algorithm | 101.52 | 80.15 | 0.00 | 0/100 | 0/0 | 100.00 |
| margin4_ccve | 99.69 | 78.53 | 0.00 | 0/100 | 0/0 | 95.00 |
| lvspec_paper | 201.23 | 146.22 | 20.00 | 74/100 | 76/0 | 100.00 |
| lvspec_adapted | 98.63 | 78.29 | 0.00 | 0/100 | 0/0 | 0.00 |

BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：ar 1.500×，Holm p=0.0012；vispec 1.005×，Holm p=1.0000；strict_shared 1.002×，Holm p=1.0000；margin4_ccve 0.996×，Holm p=1.0000；lvspec_paper 1.256×，Holm p=0.0012；lvspec_adapted 1.002×，Holm p=1.0000。

token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。

## ScienceQA image subset

长输出开发选择：vispec；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能称为成功宽松策略。

| 方法 | 内容token/s [95%CI] | 原生Draft MAT | 有效Draft MAT | 1+有效MAT | 模型请求ms [95%CI] | 最大allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 36.56 [36.31, 36.79] | NA | NA | NA | 2995.43 [2851.62, 3145.54] | 18.56 |
| vispec | 62.76 [61.35, 64.10] | 2.13 | 2.05 | 3.05 | 1756.50 [1692.38, 1823.80] | 18.56 |
| strict_shared | 63.02 [61.61, 64.35] | 2.13 | 2.05 | 3.05 | 1749.23 [1684.92, 1816.71] | 18.56 |
| bepc_algorithm | 64.52 [63.19, 65.89] | 2.23 | 2.16 | 3.16 | 1810.93 [1736.63, 1890.49] | 18.56 |
| margin4_ccve | 62.66 [61.37, 63.96] | 2.18 | 2.10 | 3.10 | 1752.47 [1681.82, 1827.93] | 18.61 |
| lvspec_paper | 96.45 [94.41, 98.21] | 3.68 | 3.65 | 4.65 | 2702.18 [2412.23, 2994.94] | 18.56 |
| lvspec_adapted | 63.06 [61.65, 64.40] | 2.13 | 2.05 | 3.05 | 1748.02 [1683.76, 1815.42] | 18.56 |

| 方法 | 最终答案Accuracy% |
| --- | ---: |
| ar | 86.00 |
| vispec | 85.00 |
| strict_shared | 85.00 |
| bepc_algorithm | 83.00 |
| margin4_ccve | 86.00 |
| lvspec_paper | 17.00 |
| lvspec_adapted | 85.00 |

| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |
|---|---:|---:|---:|---:|---:|---:|
| ar | 109.51 | 89.10 | 0.00 | 0/100 | 0/0 | 0.00 |
| vispec | 110.23 | 89.66 | 0.00 | 0/100 | 0/0 | 0.00 |
| strict_shared | 110.23 | 89.66 | 0.00 | 0/100 | 0/0 | 0.00 |
| bepc_algorithm | 116.84 | 94.95 | 0.00 | 0/100 | 0/0 | 100.00 |
| margin4_ccve | 109.81 | 89.46 | 0.00 | 0/100 | 1/0 | 87.00 |
| lvspec_paper | 260.63 | 203.45 | 27.00 | 78/100 | 78/0 | 100.00 |
| lvspec_adapted | 110.23 | 89.66 | 0.00 | 0/100 | 0/0 | 0.00 |

BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：ar 1.654×，Holm p=0.0012；vispec 0.970×，Holm p=0.2607；strict_shared 0.966×，Holm p=0.2607；margin4_ccve 0.968×，Holm p=0.2607；lvspec_paper 1.492×，Holm p=0.0012；lvspec_adapted 0.965×，Holm p=0.2607。

token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。

## COCO Caption subset

长输出开发选择：vispec；lambda=0.7，disable_relaxation=True。这是显式Strict fallback，不能称为成功宽松策略。

开发质量门预先固定为SPICE≥0.95×Strict且CHAIRs≤Strict。CIDEr-D完整保留但不参与开发门：短参考的sigma6长度惩罚可将长段落分数压到极小值，不能用绝对epsilon将它视为相等。这里的门仍只是20图自动代理，不能证明总体或人工质量非劣效。

| 方法 | 内容token/s [95%CI] | 原生Draft MAT | 有效Draft MAT | 1+有效MAT | 模型请求ms [95%CI] | 最大allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ar | 37.52 [37.41, 37.63] | NA | NA | NA | 4197.17 [4091.26, 4303.31] | 18.32 |
| vispec | 67.41 [66.78, 68.03] | 2.16 | 2.13 | 3.13 | 2332.30 [2276.39, 2388.81] | 18.31 |
| strict_shared | 67.56 [66.93, 68.19] | 2.16 | 2.13 | 3.13 | 2326.94 [2270.88, 2383.73] | 18.31 |
| bepc_algorithm | 68.89 [68.21, 69.60] | 2.27 | 2.23 | 3.23 | 2350.17 [2282.14, 2420.19] | 18.31 |
| margin4_ccve | 68.60 [67.92, 69.30] | 2.26 | 2.23 | 3.23 | 2340.14 [2277.86, 2406.74] | 18.36 |
| lvspec_paper | 89.90 [88.34, 91.53] | 3.34 | 3.30 | 4.30 | 2695.08 [2529.32, 2869.86] | 18.31 |
| lvspec_adapted | 67.61 [66.98, 68.24] | 2.16 | 2.13 | 3.13 | 2325.18 [2269.08, 2381.93] | 18.31 |

| 方法 | CIDEr-D原尺度 | SPICE% | CHAIRs% | CHAIRi% |
| --- | ---: | ---: | ---: | ---: |
| ar | 4.738859e-47 | 13.59 | 26.00 | 5.34 |
| vispec | 1.952307e-43 | 13.41 | 27.00 | 5.58 |
| strict_shared | 1.952307e-43 | 13.41 | 27.00 | 5.58 |
| bepc_algorithm | 2.498964e-51 | 13.57 | 22.00 | 5.63 |
| margin4_ccve | 2.771875e-48 | 13.94 | 21.00 | 4.72 |
| lvspec_paper | 1.628000e-10 | 11.58 | 23.00 | 5.35 |
| lvspec_adapted | 1.952307e-43 | 13.41 | 27.00 | 5.58 |

| 方法 | 可见token/响应 | 输出词/响应 | 触顶% | 答案提取失败/题 | 全部invalid/uncertain | 含committed mismatch响应% |
|---|---:|---:|---:|---:|---:|---:|
| ar | 157.48 | 132.04 | 0.00 | NA | 0/0 | 0.00 |
| vispec | 157.21 | 131.37 | 0.00 | NA | 0/0 | 0.00 |
| strict_shared | 157.21 | 131.37 | 0.00 | NA | 0/0 | 0.00 |
| bepc_algorithm | 161.91 | 135.66 | 0.00 | NA | 0/0 | 100.00 |
| margin4_ccve | 160.53 | 134.34 | 0.00 | NA | 0/0 | 100.00 |
| lvspec_paper | 242.28 | 183.52 | 8.00 | NA | 0/0 | 100.00 |
| lvspec_adapted | 157.21 | 131.37 | 0.00 | NA | 0/0 | 0.00 |

BEPC相对其余六方法的实际请求时延比（>1为BEPC快）：ar 1.786×，Holm p=0.0012；vispec 0.992×，Holm p=1.0000；strict_shared 0.990×，Holm p=1.0000；margin4_ccve 0.996×，Holm p=1.0000；lvspec_paper 1.147×，Holm p=0.0030；lvspec_adapted 0.989×，Holm p=1.0000。

token/s提高不必然代表每题更快：宽松方法可能生成更长文本或更容易触顶。请将token/s、整题时延、长度、截断与质量一起比较；接受长度本身不能证明用户等待时间缩短。

## 统计方法与结论边界

- 每问题只有一次测量，在image/group上配对bootstrap5000次。MME在category内保留完整两问group重采样。时延与吞吐CI仅反映这些图像的抽样变化，不能估计重复运行、GPU状态和计时噪声；质量使用唯一的repeat0。
- 主token/s=可见内容token总数/同步模型请求秒数，排除EOS；模型请求包括controller构造、视觉编码、prefill、Target/Draft及在线特征。CSV另存decode-only和含EOS completion吞吐。PIL/processor/H2D独立计时且方法间复用，因此不是完整客户端E2E。
- 原生Draft MAT=全部原生接受Draft token总数/全部原生验证轮数，可能包含最后block在EOS或预算后的计算。有效Draft MAT按统一首EOS/512预算裁切；另报1+有效MAT表示含每轮root的推进长度。三者均为pooled轮数比率，AR全部NA；不平均每请求MAT。
- 所有点式95%CI条件于当前子集/硬件和固定官方每图质量分数。六个BEPC比较与两个LV比较分别按task/metric作Holm校正；不控制跨全部任务和指标的全局错误率。中心化paired-bootstrap p是Monte Carlo近似，不是精确随机化检验。
- QA最终答案通过冻结末行格式提取，全部格式失败、uncertain和越界保留并计0；Acc不评价解释中的事实或逻辑。TextVQA执行官方十答案leave-one-out归一化，GQA保留exact equality。MME每类100×(acc+acc_plus)求和，14类上限2800，不是完整官方split。
- COCO整段原样计算；CIDEr-D仅作辅助，以科学计数保留极小非零值。参考只有短caption，CIDEr长度惩罚、SPICE参考覆盖不足及CHAIR对象机会数随长度变化会影响比较。自动指标不构成人类事实/覆盖Ground Truth。CIDEr DF使用当前完整100图参考，bootstrap不重新拟合DF；开发与测试绝对CIDEr不可跨split比较。
- 未定义SPICE/CHAIRi保留NA，不删除事件或填0；不按输出长度、格式成功、历史曝光或预期质量筛样。词数100–160仅是提示要求，没有最小生成强制或长度有效性过滤。
- 显存最大值是观测峰值，AR含常驻未用Draft；reserved受allocator历史影响，不是独占需求。显存增量另存CSV，不执行每请求empty_cache。
- 原生与共享Strict控制公共工程收益。LVSpec是原论文相对视觉/PST在ViSpec短树及静态图像上的适配，不是原视频/H200/K10配置的完整复现。100图及20图开发代理不足以证明总体非劣效或罕见幻觉安全；未训练、融合、利用测试质量调参或选择更快重复。

依据：[冻结实验协议](../EXPERIMENT_PROTOCOL.md)、[质量与解析协议](QUALITY_PROTOCOL.md)、[MAT/长提示/AR对齐审计](../audit/MAT_AND_LONG_PROMPT_AUDIT.md)。

机器可读结果：BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv；图：BENCHMARK_PERFORMANCE.pdf（吞吐、时延、MAT）和BENCHMARK_QUALITY.pdf。全部配置、输入、结果、官方评分来源和统计SHA见audit/STATISTICAL_ANALYSIS.json及各方法SCORING_SEAL.json。


## 官方 SPICE 资源故障与恢复

首次正式评分中，LVSpec paper 的官方 SPICE/Stanford 解析器在 16GiB Java 堆上限下出现 OutOfMemoryError，跳过句子并最终退出失败；该次尝试没有产生有效的 SPICE 分数，也没有进入最终比较。原失败日志、输入和缓存已完整归档。

恢复仅将这一个官方评分进程的 Java 堆上限提高到 64GiB，保留相同 JAR、模型、线程参数、100条完整输入及参考答案，并使用全新缓存。没有重新生成 Caption，没有改变模型、阈值、样本、指标规则或任何解码计时。其他已成功评分直接复用。

恢复输出须满足100个图像ID齐全且无重复、无 OOM/忽略解析/批评分失败；输入字节、4200条推理记录、所有冻结源码配置和既有成功评分的 SHA 均独立核对。该资源调整属于处理执行故障，不是依据测试质量更改评价标准。LMDB 长字符串缓存写入警告与忽略解析分别审计，不能将两者混为一谈。

冻结评分包装器在读取相同输入的已验证结果时，仍在 provenance 中记录其计划的16GiB命令；此次结果实际由64GiB进程生成，以独立恢复审计中的真实命令和输出SHA为准。

证据：[资源恢复记录](../audit/SPICE_RESOURCE_RECOVERY.json)、[独立恢复验证](../audit/SPICE_RESOURCE_RECOVERY_VALIDATION.md)、[原失败流水线状态](../audit/SPICE_RESOURCE_PIPELINE_FAILED_STATUS.json)。
