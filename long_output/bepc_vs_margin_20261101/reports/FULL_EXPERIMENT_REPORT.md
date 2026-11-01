# BEPC 与简单 Margin：生成速度和质量完整实验报告

比较刚完成解释的双向有界 BEPC 与普通 Target Margin。两种方法均为每轮最多2个 mismatch、每题无累计上限、tau=1，相同模型、输入、原长输出 Prompt、树、EOS 与512-token预算，每题各运行一次；本轮重新测两臂时延。

ScienceQA、COCO、TextVQA 各20开发组+100测试组，正式请求共600次。沿用已有样本而非项目未见测试集。开发数据用于完整性与数值验证，没有搜索阈值或选择最佳方法。

## 本轮直接结果

- ScienceQA：BEPC 相对 Margin，token/s -2.48%，整题模型时延 +2.00%，有效MAT +0.47%；官方 Accuracy 85.00%→86.00%。
- COCO Caption：BEPC 相对 Margin，token/s -4.89%，整题模型时延 +4.46%，有效MAT -0.22%；SPICE 12.66→13.09；CHAIRs 19.00%→20.00%。
- TextVQA：BEPC 相对 Margin，token/s -0.66%，整题模型时延 +4.89%，有效MAT +0.24%；官方 Accuracy 73.10%→73.30%。

token/s、接受长度和每题完成时间分别解释；内容变长可以同时提高token/s和增加每题时延。以下配对CI仅反映当前图像组抽样，单次运行没有重复计时证据。

## 算法与对照口径

两种方法共享真正执行的 ViSpec 验证树、候选合法前缀、原生 Target logits、逐轮两个 mismatch 限制与首个最长前缀选择。Target top1为a、Draft候选为d，原生margin g=z(a)−z(d)，普通Margin直接按g≤1放行。

BEPC把同一父节点的末层RMSNorm前状态h投影到图像prefill子空间U和正交化的近期文本子空间P，候选方向q=w(a)−w(d)，用完整h对应的RMSNorm缩放读出mU、mP。U/P是表示几何代理，不是因果分离的事实/先验。mU·mP<0且两者幅度≥1e−4时，δ=0.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；按g−δ≤1接受。residual_attenuation=false，双向gamma=.5，U/P秩8/4，视觉池64，文本窗口32；无新增Target forward。

BEPC使用已有精确边界预筛及缓存，保留其真实观察器/子空间成本；Margin不构造BEPC观察器或子空间。双方都不在累计4次后退回Strict。区别是接受评分及实现该评分必需的成本，没有修改原生Target/Draft forward。

为保留历史BEPC控制器字节，继承的BEPC统计字段all_relaxed_methods_same_budget=false仍记录上轮跨臂说明；它是遗留元数据，不表示本轮两臂预算不同。本轮权威依据是冻结config和运行期response_cap_enabled=false、response_mismatch_budget=null、round_mismatch_budget=2，两臂均经路径计数核验。

## 配置与测量

- 上游ViSpec commit：`49a33ff0ca31e281b9dc8803fd6f3041c6c0e135`。
- Target：`/data/wbb/model/Qwen2.5-VL-7B-Instruct`；Draft：`/data/wbb/model/ViSpec-Qwen2.5-VL-7B-Instruct`。
- T=0、seed=0、BF16；两张L20按图像组分片，同题两方法顺序随机化；每方法单次测量，warmup排除。
- 请求时间包含视觉编码、prefill、Target/Draft解码、验证与BEPC特征；不含预先共享的PIL/processor/H2D准备，不是客户端全链路E2E。
- token/s=可见内容token总数/同步请求秒数总和，内容token排除EOS并遵守原512可见边界；decode token/s另行列出。
- 有效MAT=可见范围接受Draft token总数/验证轮数，排除每轮root；原生MAT沿用官方整块计数。
- 按每task100个物理图像/group成对bootstrap5000次；同一抽样同时用于两方法，95%百分位CI。只有一项预定方法对比/task/metric，不作跨所有指标的全局校正。

## ScienceQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 67.16 [65.91, 68.49] | 1794.50 [1715.02, 1885.08] | 77.67 | 2.42 | 2.35 | 18.56 | 19.07 |
| margin_no_response_cap | 68.87 [67.60, 70.20] | 1759.23 [1687.05, 1839.24] | 79.87 | 2.41 | 2.34 | 18.56 | 19.07 |

| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |
|---|---:|---:|---:|---:|
| bepc_no_response_cap | 86.00 [79.00, 92.00] | 0/100 | 0 | 0 |
| margin_no_response_cap | 85.00 [78.00, 92.00] | 0/100 | 0 | 0 |

| 方法 | 内容token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | >4次比例% | 每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 120.52 | 98.79 | 0.00 | 10.06 | 10.0/17.0/23.0 | 89.00 | 2 |
| margin_no_response_cap | 121.16 | 98.25 | 0.00 | 10.18 | 10.0/17.0/19.0 | 94.00 | 2 |

| BEPC−Margin指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| accuracy | 0.0100 [-0.0200, 0.0400] | 1.18 [-2.38, 5.26] | 0.7654 |
| mean_seconds | 0.0353 [-0.0211, 0.0906] | 2.00 [-1.20, 5.20] | 0.2114 |
| throughput_tokens_per_second | -1.7103 [-2.4841, -0.9641] | -2.48 [-3.58, -1.41] | 0.0002 |
| answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | NA [NA, NA] | 1.0000 |
| effective_mean_accepted_draft_length | 0.0109 [-0.0236, 0.0462] | 0.47 [-1.01, 1.98] | 0.5497 |

全部输出与分数转移（方向Margin→BEPC）：`{"margin_0_to_bepc_0": 13, "margin_0_to_bepc_1": 2, "margin_1_to_bepc_0": 1, "margin_1_to_bepc_1": 84, "outputs_changed": 87, "outputs_unchanged": 13, "score_degraded": 1, "score_improved": 2, "score_tie": 97}`。

## COCO Caption

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 74.83 [73.99, 75.67] | 2313.37 [2231.01, 2399.17] | 83.55 | 2.71 | 2.67 | 18.31 | 19.07 |
| margin_no_response_cap | 78.68 [77.85, 79.49] | 2214.54 [2131.55, 2301.58] | 88.27 | 2.72 | 2.68 | 18.31 | 19.07 |

| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |
|---|---:|---:|---:|---:|
| bepc_no_response_cap | 2.396e-39 | 13.09 [12.26, 13.97] | 20.00 [12.00, 28.00] | 5.37 |
| margin_no_response_cap | 3.956e-45 | 12.66 [11.78, 13.59] | 19.00 [12.00, 27.00] | 4.19 |

| 方法 | 内容token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | >4次比例% | 每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 173.12 | 145.77 | 0.00 | 27.34 | 27.0/37.1/60.0 | 100.00 | 2 |
| margin_no_response_cap | 174.24 | 146.48 | 0.00 | 28.19 | 27.0/40.0/52.0 | 100.00 | 2 |

| BEPC−Margin指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| SPICE | 0.0043 [-0.0006, 0.0092] | 3.38 [-0.48, 7.40] | 0.0914 |
| CHAIRs | 0.0100 [-0.0700, 0.0900] | 5.26 [-29.41, 58.85] | 0.9082 |
| mean_seconds | 0.0988 [0.0157, 0.1881] | 4.46 [0.69, 8.68] | 0.0240 |
| throughput_tokens_per_second | -3.8455 [-4.5711, -3.0839] | -4.89 [-5.80, -3.94] | 0.0002 |
| answer_extraction_failure_fraction | NA [NA, NA] | NA [NA, NA] | NA |
| effective_mean_accepted_draft_length | -0.0058 [-0.0388, 0.0286] | -0.22 [-1.44, 1.07] | 0.7345 |

全部输出与分数转移（方向Margin→BEPC）：`{"CHAIRi_decreased": 9, "CHAIRi_equal": 74, "CHAIRi_increased": 17, "CHAIRs_decreased": 7, "CHAIRs_equal": 85, "CHAIRs_increased": 8, "CIDEr_decreased": 43, "CIDEr_equal": 2, "CIDEr_increased": 55, "SPICE_decreased": 45, "SPICE_equal": 3, "SPICE_increased": 52, "outputs_changed": 98, "outputs_unchanged": 2}`。

## TextVQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 46.52 [45.27, 47.76] | 2462.96 [2374.35, 2557.57] | 75.19 | 2.45 | 2.39 | 22.56 | 34.63 |
| margin_no_response_cap | 46.82 [45.61, 48.05] | 2348.12 [2265.41, 2432.74] | 77.99 | 2.45 | 2.38 | 22.56 | 34.63 |

| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |
|---|---:|---:|---:|---:|
| bepc_no_response_cap | 73.30 [65.00, 81.00] | 3/100 | 3 | 0 |
| margin_no_response_cap | 73.10 [64.90, 80.70] | 2/100 | 2 | 0 |

| 方法 | 内容token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | >4次比例% | 每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| bepc_no_response_cap | 114.57 | 91.47 | 0.00 | 13.90 | 14.0/23.0/30.0 | 97.00 | 2 |
| margin_no_response_cap | 109.95 | 87.27 | 0.00 | 13.35 | 13.5/21.0/24.0 | 96.00 | 2 |

| BEPC−Margin指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| accuracy | 0.0020 [-0.0570, 0.0610] | 0.27 [-7.47, 8.77] | 0.9534 |
| mean_seconds | 0.1148 [0.0631, 0.1714] | 4.89 [2.68, 7.35] | 0.0002 |
| throughput_tokens_per_second | -0.3076 [-1.0828, 0.4645] | -0.66 [-2.29, 0.99] | 0.4329 |
| answer_extraction_failure_fraction | 0.0100 [-0.0200, 0.0400] | 50.00 [NA, NA] | 0.7544 |
| effective_mean_accepted_draft_length | 0.0056 [-0.0331, 0.0439] | 0.24 [-1.39, 1.84] | 0.7668 |

全部输出与分数转移（方向Margin→BEPC）：`{"outputs_changed": 91, "outputs_unchanged": 9, "score_degraded": 5, "score_improved": 6, "score_tie": 89}`。

## 质量容忍检查与结论边界

预先固定的自动代理非劣效检查：QA最多下降2个百分点；COCO同时要求BEPC的SPICE≥Margin的95%、CHAIRs增幅≤2个百分点。采用配对bootstrap单侧95%界；不把“不显著”解释为相等。阈值是沿用pilot容忍规则的研究扩展，不是官方指标定义的安全标准。

| 任务 | BEPC相对Margin自动代理门 | 单侧界 |
|---|---|---|
| scienceqa | BOUNDARY PASS | accuracy: exact paired-count one_sided_95_lower=-0.0200 (=−2/100), pass=True |
| coco | NOT ESTABLISHED | SPICE: one_sided_95_lower=0.0066, pass=True; CHAIRs: one_sided_95_upper=0.0700, pass=False |
| textvqa | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0470, pass=False |

ScienceQA 的自动非劣效界恰好等于预设 −2pp：原始FP64代码得到 −0.020000000000000018，直接比较误判为False。使用相同5000组抽样、相同线性5%分位数和整数正确题数独立重算，精确界为 −2/100，因此修正为边界通过；没有改变容忍阈值、指标、标签或推理结果。原始判定与修正来源保存在`audit/NUMERICAL_BOUNDARY_REVIEW.json`和统计清单。此边界通过不支持优越性；COCO及TextVQA质量门仍未建立。

最终交付哈希以`audit/FINAL_DELIVERY.json`为准；`audit/REPORT_DELIVERY.json`保留数值边界修正前的生成器封印。

QA只评价最终答案，不自动验证全部长解释；COCO仍未收到新的双人独立盲评，自动指标不能替代人工事实质量。原样保留所有失败、uncertain与截断；提取失败计0，不删除差例。零分歧时bootstrap可能退化，不意味着罕见质量风险为0。

## 工程成本与复现审计

FEATURE_COST_COUNTERS.csv汇总实际特征/子空间/候选统计的覆盖、总量和均值。计数器不等于单独同步GPU耗时；总请求时间与峰值显存包含真实成本。allocated是活跃张量峰值，reserved包含缓存分配器保留区，受此前请求影响；reserved不能解释为每种算法的最低显存需求。两方法都没有额外Target forward，Margin零BEPC子空间构建由运行验证审计检查。

- ScienceQA BEPC与上一轮uncapped：输出/接受路径精确一致 100/100；输入差异 0。历史时延未复用。
- COCO Caption BEPC与上一轮uncapped：输出/接受路径精确一致 100/100；输入差异 0。历史时延未复用。
- TextVQA BEPC与上一轮uncapped：输出/接受路径精确一致 100/100；输入差异 0。历史时延未复用。

CPU官方评分在全部GPU任务退出后执行，防止SPICE/PTB评分与计时争用CPU。若全量有序样本、标签、解析器、官方实现和完整原始输出文本签名完全相同，可以复用整语料官方质量评分；每次复用记录来源及SHA，不复用计时。

## 评价方法、局限与下一步

- ScienceQA使用冻结末行答案解析器和官方多选Accuracy；TextVQA使用冻结解析器与官方VQA-style归一化/leave-one-out评分。失败计0，大小写等处理沿用官方实现。
- COCO完整段落原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR。SPICE仅沿用64G Java资源上限，未改模型、实现或公式。
- 长段落与短COCO参考的长度差异使CIDEr-D可能近乎为0；其数值变大不能单独作为事实改善。SPICE依赖参考覆盖；CHAIR主要覆盖COCO对象幻觉，不覆盖全部属性、关系、动作和遗漏。
- 无完整独立图像测试、没有重复运行计时、没有人工caption充分性/事实精度审核，不能宣布一般性的质量保持或稳定加速优势。
- 两种接受策略改变后续前缀与输出长度；这是整条解码政策比较，不是单个mismatch的因果危害标注。
- 所有改变样本均保留在CHANGED_OUTPUT_PAIRS.jsonl，包含图像引用、原始Prompt、两条完整文本及分数，不按预期选择正例。
- 最终选择应同时考虑token/s、整题时延、输出长度、MAT、最终答案、对象幻觉和人工caption质量；本轮不训练分类器、不融合规则、不根据测试调参。

证据：configs/frozen_config.json、results/main/、metrics/main/、audit/STATISTICAL_ANALYSIS.json。主表见BENCHMARK_SUMMARY.csv，全部配对差值见PAIRED_COMPARISONS.csv，图见BENCHMARK_PERFORMANCE.pdf和BENCHMARK_QUALITY.pdf。
