# BEPC 去除每题四次 mismatch 上限：完整实验报告

本轮在新目录运行，保留原 BEPC、上一轮工程优化与放宽版的源码/配置/结果。仅取消每题累计最多接受 4 个 mismatch 的约束；每轮最多 2 个 mismatch、双向 gamma=0.5、tau=1、U/P 秩8/4、文本窗口32、视觉池64、冲突门、EOS 保护、Target/Draft forward、KV 与树设置均保持不变。

复用 ScienceQA、COCO、TextVQA 各20开发/100测试组；不重新抽样。T=0，原长输出提示、512-token预算，同一问题内三种方法随机执行顺序，每种方法只测一次。开发数据不用于选择新的 gamma/阈值或接受预算。

正式测试共 900 次请求；三种方法各300次。主比较使用本轮重新运行的 capped BEPC 与 Strict，历史时延仅描述性列出。

## 核心结果

- ScienceQA：uncapped 相对 cap4，token/s +3.94%，请求时延 -0.76%，有效 MAT +8.94%；最终答案 Accuracy 83.00%→86.00%。平均提交 mismatch 3.97→10.06/题，uncapped 超过4次的比例 89.0%。
- COCO Caption：uncapped 相对 cap4，token/s +8.16%，请求时延 -1.14%，有效 MAT +19.83%；SPICE 13.57→13.09；CHAIRs 22.00%→20.00%。平均提交 mismatch 4.00→27.34/题，uncapped 超过4次的比例 100.0%。
- TextVQA：uncapped 相对 cap4，token/s +7.63%，请求时延 +4.85%，有效 MAT +10.51%；最终答案 Accuracy 72.30%→73.30%。平均提交 mismatch 4.00→13.90/题，uncapped 超过4次的比例 97.0%。

token/s、MAT 和整题时延分别报告。输出长度改变时，更高 token/s 不一定意味着更快完成问题。质量不显著下降不能证明质量相等；以下非劣效门仅是预先规定的自动指标 pilot 检查。

## 方法与唯一算法干预

| 方法 | 每题累计 mismatch 上限 | 每轮上限 | BEPC 打分 |
|---|---:|---:|---|
| strict_shared | 0 | 0 | Target greedy 连续前缀 |
| bepc_cap4 | 4 | 2 | 原双向 BEPC，gamma=0.5、tau=1 |
| bepc_no_response_cap | 无 | 2 | 完全相同 |

对原生 Target–Draft Margin g，冲突有效时 δ=0.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；接受条件 g−δ≤1。仅合法选中分支的实际 mismatch 才计数；扫描兄弟分支不消耗预算。取消 response cap 不等于取消逐轮保护，也不等于接受所有 Draft。U/P 是几何代理，不是视觉/语言因果分离。

cap4 预算耗尽后可退休 BEPC observer 并进入 Strict 快速路径；uncapped 因没有累计预算，不会执行这种退休。因此 uncapped 额外特征维护的代价全部计入时延；不能把理论接受增长直接当作净加速。

## 速度、MAT、显存与生成质量

token/s=首 EOS/512 可见范围内内容 token 总数÷同步模型请求总秒数，不含EOS；模型请求包括视觉编码、prefill、控制器和Target/Draft解码。PIL/processor/H2D预处理单独记录并在方法间复用，所以不是完整客户端E2E。有效MAT=实际可见范围内接受的Draft token总数÷验证轮数，排除每轮root；原生MAT保留官方整块计数。

### ScienceQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 63.05 [61.70, 64.38] | 1748.41 [1687.04, 1814.72] | 73.15 | 2.13 | 2.05 | 18.56 |
| bepc_cap4 | 64.80 [63.50, 66.15] | 1802.96 [1727.02, 1883.56] | 74.85 | 2.23 | 2.16 | 18.56 |
| bepc_no_response_cap | 67.36 [66.12, 68.67] | 1789.22 [1709.85, 1878.45] | 77.90 | 2.42 | 2.35 | 18.56 |

| 方法 | 官方最终答案Accuracy % [95% CI] | 答案提取失败 |
|---|---:|---:|
| strict_shared | 85.00 [78.00, 92.00] | 0/100 |
| bepc_cap4 | 83.00 [75.00, 90.00] | 0/100 |
| bepc_no_response_cap | 86.00 [79.00, 92.00] | 0/100 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 超过4次% | 实际每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_shared | 110.23 | 89.66 | 0.00 | 0.00 | 0.0/0.0/0.0 | 0.00 | 0 |
| bepc_cap4 | 116.84 | 94.95 | 0.00 | 3.97 | 4.0/4.0/4.0 | 0.00 | 2 |
| bepc_no_response_cap | 120.52 | 98.79 | 0.00 | 10.06 | 10.0/17.0/23.0 | 89.00 | 2 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_no_response_cap−bepc_cap4 | accuracy | 0.0300 [0.0000, 0.0700] | 0.3905 |
| bepc_cap4−strict_shared | accuracy | -0.0200 [-0.0700, 0.0300] | 1.0000 |
| bepc_no_response_cap−strict_shared | accuracy | 0.0100 [-0.0300, 0.0500] | 1.0000 |
| bepc_no_response_cap−bepc_cap4 | mean_seconds | -0.0137 [-0.0671, 0.0380] | 0.5975 |
| bepc_cap4−strict_shared | mean_seconds | 0.0545 [-0.0105, 0.1213] | 0.3263 |
| bepc_no_response_cap−strict_shared | mean_seconds | 0.0408 [-0.0269, 0.1109] | 0.4991 |
| bepc_no_response_cap−bepc_cap4 | throughput_tokens_per_second | 2.5541 [1.9007, 3.2451] | 0.0006 |
| bepc_cap4−strict_shared | throughput_tokens_per_second | 1.7590 [0.8816, 2.6080] | 0.0006 |
| bepc_no_response_cap−strict_shared | throughput_tokens_per_second | 4.3131 [3.4613, 5.1820] | 0.0006 |
| bepc_no_response_cap−bepc_cap4 | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_cap4−strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_no_response_cap−strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_no_response_cap−bepc_cap4 | effective_mean_accepted_draft_length | 0.1927 [0.1546, 0.2326] | 0.0006 |
| bepc_cap4−strict_shared | effective_mean_accepted_draft_length | 0.1051 [0.0609, 0.1480] | 0.0006 |
| bepc_no_response_cap−strict_shared | effective_mean_accepted_draft_length | 0.2978 [0.2564, 0.3400] | 0.0006 |

uncapped 相对 cap4 的完整转移计数：`{"0_to_0": 14, "0_to_1": 3, "1_to_1": 83, "outputs_changed": 89, "outputs_unchanged": 11, "score_improved": 3, "score_tie": 97}`。

### COCO Caption

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 67.61 [66.98, 68.24] | 2325.17 [2268.24, 2382.04] | 75.38 | 2.16 | 2.13 | 18.31 |
| bepc_cap4 | 69.35 [68.64, 70.05] | 2334.54 [2267.63, 2403.73] | 77.31 | 2.27 | 2.23 | 18.31 |
| bepc_no_response_cap | 75.01 [74.18, 75.83] | 2307.85 [2225.74, 2393.66] | 83.73 | 2.71 | 2.67 | 18.31 |

| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |
|---|---:|---:|---:|---:|
| strict_shared | 1.952e-43 | 13.41 [12.53, 14.32] | 27.00 [18.00, 36.00] | 5.58 |
| bepc_cap4 | 2.499e-51 | 13.57 [12.70, 14.45] | 22.00 [14.00, 30.00] | 5.63 |
| bepc_no_response_cap | 2.396e-39 | 13.09 [12.26, 13.97] | 20.00 [12.00, 28.00] | 5.37 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 超过4次% | 实际每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_shared | 157.21 | 131.37 | 0.00 | 0.00 | 0.0/0.0/0.0 | 0.00 | 0 |
| bepc_cap4 | 161.91 | 135.66 | 0.00 | 4.00 | 4.0/4.0/4.0 | 0.00 | 2 |
| bepc_no_response_cap | 173.12 | 145.77 | 0.00 | 27.34 | 27.0/37.1/60.0 | 100.00 | 2 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_no_response_cap−bepc_cap4 | SPICE | -0.0048 [-0.0108, 0.0010] | 0.3323 |
| bepc_cap4−strict_shared | SPICE | 0.0016 [-0.0055, 0.0085] | 0.7898 |
| bepc_no_response_cap−strict_shared | SPICE | -0.0032 [-0.0104, 0.0041] | 0.7898 |
| bepc_no_response_cap−bepc_cap4 | CHAIRs | -0.0200 [-0.0900, 0.0500] | 0.6589 |
| bepc_cap4−strict_shared | CHAIRs | -0.0500 [-0.1200, 0.0200] | 0.4159 |
| bepc_no_response_cap−strict_shared | CHAIRs | -0.0700 [-0.1500, 0.0200] | 0.3965 |
| bepc_no_response_cap−bepc_cap4 | mean_seconds | -0.0267 [-0.1019, 0.0531] | 1.0000 |
| bepc_cap4−strict_shared | mean_seconds | 0.0094 [-0.0554, 0.0699] | 1.0000 |
| bepc_no_response_cap−strict_shared | mean_seconds | -0.0173 [-0.0947, 0.0600] | 1.0000 |
| bepc_no_response_cap−bepc_cap4 | throughput_tokens_per_second | 5.6596 [4.8535, 6.4854] | 0.0006 |
| bepc_cap4−strict_shared | throughput_tokens_per_second | 1.7418 [1.0464, 2.4366] | 0.0006 |
| bepc_no_response_cap−strict_shared | throughput_tokens_per_second | 7.4014 [6.5861, 8.2399] | 0.0006 |
| bepc_no_response_cap−bepc_cap4 | effective_mean_accepted_draft_length | 0.4426 [0.4049, 0.4807] | 0.0006 |
| bepc_cap4−strict_shared | effective_mean_accepted_draft_length | 0.0995 [0.0651, 0.1345] | 0.0006 |
| bepc_no_response_cap−strict_shared | effective_mean_accepted_draft_length | 0.5421 [0.5043, 0.5801] | 0.0006 |

uncapped 相对 cap4 的完整转移计数：`{"CHAIRi_decreased": 13, "CHAIRi_equal": 78, "CHAIRi_increased": 9, "CHAIRs_decreased": 7, "CHAIRs_equal": 88, "CHAIRs_increased": 5, "CIDEr_decreased": 62, "CIDEr_increased": 38, "SPICE_decreased": 53, "SPICE_increased": 47, "outputs_changed": 100}`。

### TextVQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 41.90 [40.71, 43.08] | 2354.10 [2272.67, 2440.59] | 69.58 | 2.08 | 2.01 | 22.56 |
| bepc_cap4 | 43.32 [42.15, 44.50] | 2343.31 [2264.43, 2424.64] | 72.21 | 2.23 | 2.16 | 22.56 |
| bepc_no_response_cap | 46.63 [45.37, 47.87] | 2457.07 [2368.63, 2551.63] | 75.39 | 2.45 | 2.39 | 22.56 |

| 方法 | 官方最终答案Accuracy % [95% CI] | 答案提取失败 |
|---|---:|---:|
| strict_shared | 71.60 [63.30, 79.30] | 0/100 |
| bepc_cap4 | 72.30 [64.10, 79.80] | 0/100 |
| bepc_no_response_cap | 73.30 [65.00, 81.00] | 3/100 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 超过4次% | 实际每轮max |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_shared | 98.63 | 78.29 | 0.00 | 0.00 | 0.0/0.0/0.0 | 0.00 | 0 |
| bepc_cap4 | 101.52 | 80.15 | 0.00 | 4.00 | 4.0/4.0/4.0 | 0.00 | 2 |
| bepc_no_response_cap | 114.57 | 91.47 | 0.00 | 13.90 | 14.0/23.0/30.0 | 97.00 | 2 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_no_response_cap−bepc_cap4 | accuracy | 0.0100 [-0.0300, 0.0510] | 1.0000 |
| bepc_cap4−strict_shared | accuracy | 0.0070 [-0.0380, 0.0530] | 1.0000 |
| bepc_no_response_cap−strict_shared | accuracy | 0.0170 [-0.0400, 0.0770] | 1.0000 |
| bepc_no_response_cap−bepc_cap4 | mean_seconds | 0.1138 [0.0498, 0.1796] | 0.0024 |
| bepc_cap4−strict_shared | mean_seconds | -0.0108 [-0.0671, 0.0438] | 0.7133 |
| bepc_no_response_cap−strict_shared | mean_seconds | 0.1030 [0.0295, 0.1792] | 0.0136 |
| bepc_no_response_cap−bepc_cap4 | throughput_tokens_per_second | 3.3053 [2.4133, 4.2086] | 0.0006 |
| bepc_cap4−strict_shared | throughput_tokens_per_second | 1.4263 [0.6435, 2.2132] | 0.0012 |
| bepc_no_response_cap−strict_shared | throughput_tokens_per_second | 4.7316 [3.8079, 5.6382] | 0.0006 |
| bepc_no_response_cap−bepc_cap4 | answer_extraction_failure_fraction | 0.0300 [0.0000, 0.0700] | 0.3917 |
| bepc_cap4−strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_no_response_cap−strict_shared | answer_extraction_failure_fraction | 0.0300 [0.0000, 0.0700] | 0.3917 |
| bepc_no_response_cap−bepc_cap4 | effective_mean_accepted_draft_length | 0.2268 [0.1822, 0.2728] | 0.0006 |
| bepc_cap4−strict_shared | effective_mean_accepted_draft_length | 0.1511 [0.1003, 0.1981] | 0.0006 |
| bepc_no_response_cap−strict_shared | effective_mean_accepted_draft_length | 0.3779 [0.3188, 0.4326] | 0.0006 |

uncapped 相对 cap4 的完整转移计数：`{"outputs_changed": 97, "outputs_unchanged": 3, "score_degraded": 2, "score_improved": 4, "score_tie": 94}`。

## 预先规定的质量容忍检查

QA允许最多2个百分点下降；COCO同时要求SPICE不少于对照95%、CHAIRs增幅不超过2个百分点。使用配对bootstrap单侧95%界，既比较uncapped/cap4，也分别比较两种BEPC/Strict；不能以p>0.05替代非劣效。

| 任务 | 方法/对照 | 自动代理门 | 单侧界 |
|---|---|---|---|
| scienceqa | bepc_no_response_cap/bepc_cap4 | PASS | accuracy: one_sided_95_lower=0.0100, pass=True |
| scienceqa | bepc_cap4/strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0600, pass=False |
| scienceqa | bepc_no_response_cap/strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0300, pass=False |
| coco | bepc_no_response_cap/bepc_cap4 | NOT ESTABLISHED | SPICE: one_sided_95_lower=-0.0029, pass=False; CHAIRs: one_sided_95_upper=0.0400, pass=False |
| coco | bepc_cap4/strict_shared | PASS | SPICE: one_sided_95_lower=0.0028, pass=True; CHAIRs: one_sided_95_upper=0.0100, pass=True |
| coco | bepc_no_response_cap/strict_shared | NOT ESTABLISHED | SPICE: one_sided_95_lower=-0.0023, pass=False; CHAIRs: one_sided_95_upper=0.0000, pass=True |
| textvqa | bepc_no_response_cap/bepc_cap4 | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0240, pass=False |
| textvqa | bepc_cap4/strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0300, pass=False |
| textvqa | bepc_no_response_cap/strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0320, pass=False |

所有门仅限当前100题/任务固定pilot自动评价；零分歧时非参数bootstrap可能退化，不能证明总体罕见退化率为0。COCO没有新增双人盲评，不能把这些指标称为人工Ground Truth，也不能给出生成事实安全保证。

## 额外特征成本与历史一致性

FEATURE_COST_COUNTERS.csv保存每个数值计数器（含子空间、full-score、视觉预筛、observer构建/退休）的覆盖、总数、均值与最大值。这些是诊断计数，不将CPU enqueue耗时当作独立同步GPU feature时延。总请求/decode时间与显存已包含全部实际开销。

- ScienceQA strict_shared 与上轮 strict_shared：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。
- ScienceQA bepc_cap4 与上轮 bepc_fast：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。
- COCO Caption strict_shared 与上轮 strict_shared：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。
- COCO Caption bepc_cap4 与上轮 bepc_fast：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。
- TextVQA strict_shared 与上轮 strict_shared：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。
- TextVQA bepc_cap4 与上轮 bepc_fast：输出/路径字段一致 100/100；输入差异0。所有差异保留，不擅自归因为BF16/KV/mask。

## 与上轮工程优化实验的描述性对照

以下历史速度来自不同时段，不参与本轮配对检验，也不用于选择最佳算法。原长提示、数据、生成预算相同。

| 任务 | 上轮方法 | token/s | 请求ms | 有效MAT | 主质量 |
|---|---|---:|---:|---:|---|
| scienceqa | strict_shared | 63.05 | 1748.32 | 2.05 | 85.00% |
| scienceqa | bepc_original | 64.55 | 1810.05 | 2.16 | 83.00% |
| scienceqa | bepc_fast | 64.76 | 1804.25 | 2.16 | 83.00% |
| scienceqa | bepc_rescue0625 | 64.69 | 1800.49 | 2.16 | 83.00% |
| coco | strict_shared | 67.64 | 2324.35 | 2.13 | SPICE 13.41 / CHAIRs 27.00 |
| coco | bepc_original | 68.98 | 2347.37 | 2.23 | SPICE 13.57 / CHAIRs 22.00 |
| coco | bepc_fast | 69.39 | 2333.49 | 2.23 | SPICE 13.57 / CHAIRs 22.00 |
| coco | bepc_rescue0625 | 69.49 | 2336.16 | 2.24 | SPICE 13.56 / CHAIRs 22.00 |
| textvqa | strict_shared | 41.93 | 2352.45 | 2.01 | 71.60% |
| textvqa | bepc_original | 43.26 | 2346.61 | 2.16 | 72.30% |
| textvqa | bepc_fast | 43.38 | 2340.24 | 2.16 | 72.30% |
| textvqa | bepc_rescue0625 | 43.38 | 2343.90 | 2.16 | 72.90% |

## 评价实现、统计与局限

- ScienceQA沿用官方多选Accuracy和冻结末行答案解析器；TextVQA沿用官方VQA-style leave-one-out归一化。提取失败和歧义保留并计0；只评最终答案，不审计长解释。
- COCO整段输出原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR。SPICE沿用已验证的64G资源上限、新缓存，未改scorer/模型/公式。全量有序样本、标签、原文与签名逐字相同时才复用已有完整语料评分；不复用速度。
- 原始解析失败、生成截断和所有正负改变样本均保留；CHANGED_OUTPUT_PAIRS.jsonl包含全部改变输出的图像引用、原文及分数，不能把自动分数变化直接等同具体事实伤害。
- 图像/group配对bootstrap5000次，三个预先对比按task/metric做Holm校正，不控制全部任务/指标全局错误率；CI只量化图像抽样，不量化单次计时运行噪声。
- 当前样本沿用旧实验，没有项目未见独立性；没有根据输出、标签或是否变好重新抽样，也没有根据测试结果调gamma/预算/阈值。
- 长描述相对短参考使CIDEr极小；SPICE受参考覆盖限制，CHAIR只覆盖COCO80对象幻觉，不覆盖全部属性/关系/动作/遗漏。
- 取消累计预算会改变后续前缀和可见事件，是完整解码政策比较；不将所有新增mismatch称为单事件因果效应，也不训练新的验证器。
- 采用uncapped必须综合整题时延、token/s、MAT、输出长度、最终答案、对象幻觉和人工质量。当前固定pilot无法承诺所有输入上更快且保质。

证据入口：configs/frozen_config.json、results/main/、metrics/main/、audit/STATISTICAL_ANALYSIS.json。图表见BENCHMARK_PERFORMANCE.pdf、BENCHMARK_QUALITY.pdf；主表与配对差值分别为BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv。

## 最终执行与操纵核验

旧代码、前两轮结果的213项SHA在运行前后均一致。180次开发与900次主测试全部保留；保留上限BEPC和Strict的输入、原始/可见token及选中路径均与上轮一致。本轮CPU官方评分仅在两个GPU生成进程都结束后运行，无官方评分与GPU计时重叠。

| 任务 | uncapped实际超过4次/100 | 累计mismatch均值 | 最大值 | 首个选中路径差异已核验题数 |
|---|---:|---:|---:|---:|
| scienceqa | 89 | 10.06 | 23 | 89 |
| coco | 100 | 27.34 | 60 | 100 |
| textvqa | 97 | 13.90 | 30 | 97 |

在首个选中路径发生差异前，两策略采用相同前缀。核验要求差异出现时共享累计mismatch至少为3：累计不超过2时，一轮至多2次新接受不会碰到整题4次限制，因此两策略不应提前分叉。这项检查支持干预位置正确，不将整条轨迹的质量差异当作单事件Harmful因果标签。

CIDEr的bootstrap固定完整参考语料DF与官方逐图分数，并非每次重采样重建DF。自动指标与QA最终答案只覆盖各自评价范围，人工图像条件事实/充分性评价仍未提供。

本报告及全部当前文件的最终哈希以`audit/FINAL_DELIVERY.json`为准；`audit/REPORT_DELIVERY.json`保留补充章节前的生成器阶段封印，来源关系记录在REPORT_ADDENDUM_PROVENANCE。
