# BEPC 独立代码优化与定向放宽：三任务完整实验报告

本轮保留原 BEPC 代码、原五任务实验配置和结果，另建独立源码目录。复用上轮 ScienceQA、COCO、TextVQA 各 20 组开发、100 组测试图像及原长输出提示，T=0、512-token 上限、Draft 树、EOS、KV 更新与每轮/每题 2/4 mismatch 预算不变。原版 BEPC、工程优化版和单向小幅增强 rescue 版在同轮重新计时；历史七方法仅作描述性对照。

主测试共 1200 次请求，四个方法各 300 次；每问题每方法只测一次。工程版与本轮原版在 300/300 题的全部冻结输出/接受路径字段完全一致。这验证当前样本上的工程等价性，不能代替对全部输入的形式证明。

## 核心结论与边界

- ScienceQA：工程版 token/s 相对原版变化 +0.32%，整题请求时延变化 -0.32%；工程版逐字段一致 100/100。rescue 版为 最终答案 Accuracy 83.00%，改变输出 3/100 题。
- COCO Caption：工程版 token/s 相对原版变化 +0.59%，整题请求时延变化 -0.59%；工程版逐字段一致 100/100。rescue 版为 SPICE 13.56%、CHAIRs 22.00%，改变输出 5/100 题。
- TextVQA：工程版 token/s 相对原版变化 +0.27%，整题请求时延变化 -0.27%；工程版逐字段一致 100/100。rescue 版为 最终答案 Accuracy 72.90%，改变输出 12/100 题。

这些结果同时保留增益与退化。token/s、MAT 增加不代表整题更快，质量差异不显著也不能推出质量相等。所有 CI 仅反映图像抽样变化；一次计时无法量化运行间噪声。COCO 未新增人工盲评，自动指标通过仍不能称为事实安全保证。

## 方法与冻结的改动

| 方法 | 接受规则/改动 |
|---|---|
| strict_shared | 同一公共工程路径的 T=0 Strict；仅接受 Target greedy token |
| bepc_original | 原双向 BEPC，tau=1、gamma=0.5、rank U/P=8/4、历史文本窗口32、视觉池64、wR=1 |
| bepc_fast | 保持同一打分与决策；独立代码实现工程优化，具体实现和校验见本目录协议/审计 |
| bepc_rescue0625 | 与工程版相同，只将视觉支持 Draft 且文本支持 Target 的 rescue 幅度 0.5→0.625；protection=0.5 |

记 Target–Draft 原生 Margin 为 g，视觉/文本正交子空间候选读出为 mU、mP。冲突门、有效性门、min 截断、tau=1 和预算均不变：rescue 分支 delta=+0.625·min(|mU|,|mP|,1)，protection 分支 delta=-0.5·min(|mU|,|mP|,1)，其余 delta=0；接受条件 g-delta≤1。因此 rescue 理论 Margin 上界从 1.5 提至 1.625，不是所有 g≤1.625 一律放行。U/P 是几何代理，不是纯视觉证据/纯语言先验的因果分解。

本轮没有训练分类器、融合新信号、改 Strict 内核、增加 Target forward 或从测试质量挑阈值；测试集及开发集沿用旧样本，不再具有项目未见性质。开发与测试图像仍按冻结清单分离。

## 本轮配对速度、接受长度与显存

token/s=可见内容 token 总数/同步模型请求秒数，EOS 不计；请求包含视觉编码、prefill、controller 和 Target/Draft 解码。PIL/processor/H2D 预处理单独计时并在方法间复用，故不是完整客户端 E2E。有效 MAT=首 EOS/512 预算内实际接受的 Draft token 总数/验证轮数，排除每轮 root；不对每请求 MAT 求平均。

### ScienceQA

| 方法 | 内容 token/s [95% CI] | 请求 ms [95% CI] | 原生 MAT | 有效 MAT | 含 root 推进量 | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 63.05 [61.70, 64.38] | 1748.32 [1687.14, 1814.58] | 2.13 | 2.05 | 3.05 | 18.56 |
| bepc_original | 64.55 [63.25, 65.89] | 1810.05 [1733.65, 1891.37] | 2.23 | 2.16 | 3.16 | 18.56 |
| bepc_fast | 64.76 [63.44, 66.11] | 1804.25 [1728.70, 1884.92] | 2.23 | 2.16 | 3.16 | 18.56 |
| bepc_rescue0625 | 64.69 [63.37, 66.06] | 1800.49 [1726.67, 1879.78] | 2.23 | 2.16 | 3.16 | 18.56 |

| 方法 | 官方最终答案 Accuracy % [95% CI] | 答案提取失败 |
|---|---:|---:|
| strict_shared | 85.00 [78.00, 92.00] | 0/100 |
| bepc_original | 83.00 [75.00, 90.00] | 0/100 |
| bepc_fast | 83.00 [75.00, 90.00] | 0/100 |
| bepc_rescue0625 | 83.00 [75.00, 90.00] | 0/100 |

| 方法 | token/题 | 词/题 | 截断 % | committed mismatch/题 | mismatch 响应 % |
|---|---:|---:|---:|---:|---:|
| strict_shared | 110.23 | 89.66 | 0.00 | 0.00 | 0.00 |
| bepc_original | 116.84 | 94.95 | 0.00 | 3.97 | 100.00 |
| bepc_fast | 116.84 | 94.95 | 0.00 | 3.97 | 100.00 |
| bepc_rescue0625 | 116.48 | 94.72 | 0.00 | 3.97 | 100.00 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_fast − bepc_original | accuracy | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | accuracy | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_original − strict_shared | accuracy | -0.0200 [-0.0700, 0.0300] | 1.0000 |
| bepc_rescue0625 − strict_shared | accuracy | -0.0200 [-0.0700, 0.0300] | 1.0000 |
| bepc_fast − bepc_original | mean_seconds | -0.0058 [-0.0077, -0.0028] | 0.0032 |
| bepc_rescue0625 − bepc_fast | mean_seconds | -0.0038 [-0.0170, 0.0045] | 0.6145 |
| bepc_original − strict_shared | mean_seconds | 0.0617 [-0.0036, 0.1291] | 0.2136 |
| bepc_rescue0625 − strict_shared | mean_seconds | 0.0522 [-0.0124, 0.1172] | 0.2420 |
| bepc_fast − bepc_original | throughput_tokens_per_second | 0.2074 [0.1016, 0.2729] | 0.0024 |
| bepc_rescue0625 − bepc_fast | throughput_tokens_per_second | -0.0645 [-0.2277, 0.1280] | 0.4583 |
| bepc_original − strict_shared | throughput_tokens_per_second | 1.5016 [0.6144, 2.3599] | 0.0028 |
| bepc_rescue0625 − strict_shared | throughput_tokens_per_second | 1.6444 [0.7771, 2.4713] | 0.0008 |
| bepc_fast − bepc_original | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_original − strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_fast − bepc_original | effective_mean_accepted_draft_length | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | effective_mean_accepted_draft_length | 0.0005 [-0.0014, 0.0029] | 1.0000 |
| bepc_original − strict_shared | effective_mean_accepted_draft_length | 0.1051 [0.0609, 0.1480] | 0.0008 |
| bepc_rescue0625 − strict_shared | effective_mean_accepted_draft_length | 0.1056 [0.0615, 0.1486] | 0.0008 |

放宽版相对工程版的全部转移统计：`{"0_to_0": 17, "1_to_1": 83, "outputs_changed": 3, "outputs_unchanged": 97, "score_tie": 100}`。所有改变的 question_id、image/group_id 和分数差值完整保存在 audit/STATISTICAL_ANALYSIS.json；没有按正负结果筛选。

### COCO Caption

| 方法 | 内容 token/s [95% CI] | 请求 ms [95% CI] | 原生 MAT | 有效 MAT | 含 root 推进量 | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 67.64 [67.00, 68.26] | 2324.35 [2267.49, 2381.42] | 2.16 | 2.13 | 3.13 | 18.31 |
| bepc_original | 68.98 [68.27, 69.67] | 2347.37 [2280.00, 2416.83] | 2.27 | 2.23 | 3.23 | 18.31 |
| bepc_fast | 69.39 [68.68, 70.08] | 2333.49 [2266.57, 2402.45] | 2.27 | 2.23 | 3.23 | 18.31 |
| bepc_rescue0625 | 69.49 [68.75, 70.23] | 2336.16 [2269.93, 2405.54] | 2.27 | 2.24 | 3.24 | 18.31 |

| 方法 | CIDEr-D 原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |
|---|---:|---:|---:|---:|
| strict_shared | 1.952e-43 | 13.41 [12.53, 14.32] | 27.00 [18.00, 36.00] | 5.58 |
| bepc_original | 2.499e-51 | 13.57 [12.70, 14.45] | 22.00 [14.00, 30.00] | 5.63 |
| bepc_fast | 2.499e-51 | 13.57 [12.70, 14.45] | 22.00 [14.00, 30.00] | 5.63 |
| bepc_rescue0625 | 2.499e-51 | 13.56 [12.70, 14.43] | 22.00 [14.00, 30.00] | 5.48 |

| 方法 | token/题 | 词/题 | 截断 % | committed mismatch/题 | mismatch 响应 % |
|---|---:|---:|---:|---:|---:|
| strict_shared | 157.21 | 131.37 | 0.00 | 0.00 | 0.00 |
| bepc_original | 161.91 | 135.66 | 0.00 | 4.00 | 100.00 |
| bepc_fast | 161.91 | 135.66 | 0.00 | 4.00 | 100.00 |
| bepc_rescue0625 | 162.34 | 136.03 | 0.00 | 4.00 | 100.00 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_fast − bepc_original | SPICE | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | SPICE | -0.0002 [-0.0026, 0.0018] | 1.0000 |
| bepc_original − strict_shared | SPICE | 0.0016 [-0.0055, 0.0085] | 1.0000 |
| bepc_rescue0625 − strict_shared | SPICE | 0.0014 [-0.0053, 0.0081] | 1.0000 |
| bepc_fast − bepc_original | CHAIRs | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | CHAIRs | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_original − strict_shared | CHAIRs | -0.0500 [-0.1200, 0.0200] | 0.8318 |
| bepc_rescue0625 − strict_shared | CHAIRs | -0.0500 [-0.1200, 0.0200] | 0.8318 |
| bepc_fast − bepc_original | mean_seconds | -0.0139 [-0.0145, -0.0133] | 0.0008 |
| bepc_rescue0625 − bepc_fast | mean_seconds | 0.0027 [-0.0090, 0.0148] | 1.0000 |
| bepc_original − strict_shared | mean_seconds | 0.0230 [-0.0419, 0.0838] | 1.0000 |
| bepc_rescue0625 − strict_shared | mean_seconds | 0.0118 [-0.0519, 0.0743] | 1.0000 |
| bepc_fast − bepc_original | throughput_tokens_per_second | 0.4101 [0.3988, 0.4207] | 0.0008 |
| bepc_rescue0625 − bepc_fast | throughput_tokens_per_second | 0.1049 [-0.0353, 0.2896] | 0.2118 |
| bepc_original − strict_shared | throughput_tokens_per_second | 1.3392 [0.6525, 2.0350] | 0.0012 |
| bepc_rescue0625 − strict_shared | throughput_tokens_per_second | 1.8541 [1.1285, 2.5814] | 0.0008 |
| bepc_fast − bepc_original | effective_mean_accepted_draft_length | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | effective_mean_accepted_draft_length | 0.0060 [-0.0006, 0.0151] | 0.2767 |
| bepc_original − strict_shared | effective_mean_accepted_draft_length | 0.0995 [0.0651, 0.1345] | 0.0008 |
| bepc_rescue0625 − strict_shared | effective_mean_accepted_draft_length | 0.1054 [0.0698, 0.1420] | 0.0008 |

放宽版相对工程版的全部转移统计：`{"CHAIRi_decreased": 1, "CHAIRi_equal": 99, "CHAIRs_equal": 100, "CIDEr_decreased": 3, "CIDEr_equal": 95, "CIDEr_increased": 2, "SPICE_decreased": 1, "SPICE_equal": 95, "SPICE_increased": 4, "outputs_changed": 5, "outputs_unchanged": 95}`。所有改变的 question_id、image/group_id 和分数差值完整保存在 audit/STATISTICAL_ANALYSIS.json；没有按正负结果筛选。

### TextVQA

| 方法 | 内容 token/s [95% CI] | 请求 ms [95% CI] | 原生 MAT | 有效 MAT | 含 root 推进量 | 峰值 allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| strict_shared | 41.93 [40.74, 43.10] | 2352.45 [2271.13, 2439.20] | 2.08 | 2.01 | 3.01 | 22.56 |
| bepc_original | 43.26 [42.09, 44.45] | 2346.61 [2267.90, 2427.57] | 2.23 | 2.16 | 3.16 | 22.56 |
| bepc_fast | 43.38 [42.20, 44.56] | 2340.24 [2261.86, 2421.42] | 2.23 | 2.16 | 3.16 | 22.56 |
| bepc_rescue0625 | 43.38 [42.22, 44.54] | 2343.90 [2269.77, 2420.66] | 2.23 | 2.16 | 3.16 | 22.56 |

| 方法 | 官方最终答案 Accuracy % [95% CI] | 答案提取失败 |
|---|---:|---:|
| strict_shared | 71.60 [63.30, 79.30] | 0/100 |
| bepc_original | 72.30 [64.10, 79.80] | 0/100 |
| bepc_fast | 72.30 [64.10, 79.80] | 0/100 |
| bepc_rescue0625 | 72.90 [64.90, 80.40] | 0/100 |

| 方法 | token/题 | 词/题 | 截断 % | committed mismatch/题 | mismatch 响应 % |
|---|---:|---:|---:|---:|---:|
| strict_shared | 98.63 | 78.29 | 0.00 | 0.00 | 0.00 |
| bepc_original | 101.52 | 80.15 | 0.00 | 4.00 | 100.00 |
| bepc_fast | 101.52 | 80.15 | 0.00 | 4.00 | 100.00 |
| bepc_rescue0625 | 101.69 | 80.32 | 0.00 | 4.00 | 100.00 |

| 对比 | 指标 | 差值 [95% CI] | Holm p |
|---|---|---:|---:|
| bepc_fast − bepc_original | accuracy | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | accuracy | 0.0060 [0.0000, 0.0180] | 1.0000 |
| bepc_original − strict_shared | accuracy | 0.0070 [-0.0380, 0.0530] | 1.0000 |
| bepc_rescue0625 − strict_shared | accuracy | 0.0130 [-0.0310, 0.0590] | 1.0000 |
| bepc_fast − bepc_original | mean_seconds | -0.0064 [-0.0071, -0.0056] | 0.0008 |
| bepc_rescue0625 − bepc_fast | mean_seconds | 0.0037 [-0.0171, 0.0257] | 1.0000 |
| bepc_original − strict_shared | mean_seconds | -0.0058 [-0.0626, 0.0490] | 1.0000 |
| bepc_rescue0625 − strict_shared | mean_seconds | -0.0085 [-0.0663, 0.0476] | 1.0000 |
| bepc_fast − bepc_original | throughput_tokens_per_second | 0.1177 [0.1037, 0.1313] | 0.0008 |
| bepc_rescue0625 − bepc_fast | throughput_tokens_per_second | 0.0048 [-0.4328, 0.4105] | 0.9820 |
| bepc_original − strict_shared | throughput_tokens_per_second | 1.3359 [0.5582, 2.1222] | 0.0032 |
| bepc_rescue0625 − strict_shared | throughput_tokens_per_second | 1.4584 [0.6788, 2.2362] | 0.0024 |
| bepc_fast − bepc_original | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_original − strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − strict_shared | answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_fast − bepc_original | effective_mean_accepted_draft_length | 0.0000 [0.0000, 0.0000] | 1.0000 |
| bepc_rescue0625 − bepc_fast | effective_mean_accepted_draft_length | -0.0006 [-0.0180, 0.0169] | 1.0000 |
| bepc_original − strict_shared | effective_mean_accepted_draft_length | 0.1511 [0.1003, 0.1981] | 0.0008 |
| bepc_rescue0625 − strict_shared | effective_mean_accepted_draft_length | 0.1505 [0.1026, 0.1953] | 0.0008 |

放宽版相对工程版的全部转移统计：`{"outputs_changed": 12, "outputs_unchanged": 88, "score_improved": 1, "score_tie": 99}`。所有改变的 question_id、image/group_id 和分数差值完整保存在 audit/STATISTICAL_ANALYSIS.json；没有按正负结果筛选。

## 事先规定的质量容忍检查

QA 容忍最多 2 个百分点下降；COCO 要求 SPICE 不低于对照的 95%，CHAIRs 增幅不超过 2 个百分点。这里用配对 bootstrap 的单侧 95% 界作 pilot 自动代理检查，不将“p>0.05”当作非劣效。COCO 两门必须同时通过。所有界都在当前固定样本和自动指标条件下解释。

| 任务 | 方法 / 对照 | 自动代理门 | 逐指标单侧界 |
|---|---|---|---|
| scienceqa | bepc_fast / bepc_original | PASS | accuracy: one_sided_95_lower=0.0000, pass=True |
| scienceqa | bepc_rescue0625 / bepc_fast | PASS | accuracy: one_sided_95_lower=0.0000, pass=True |
| scienceqa | bepc_original / strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0600, pass=False |
| scienceqa | bepc_rescue0625 / strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0600, pass=False |
| coco | bepc_fast / bepc_original | PASS | SPICE: one_sided_95_lower=0.0064, pass=True; CHAIRs: one_sided_95_upper=0.0000, pass=True |
| coco | bepc_rescue0625 / bepc_fast | PASS | SPICE: one_sided_95_lower=0.0047, pass=True; CHAIRs: one_sided_95_upper=0.0000, pass=True |
| coco | bepc_original / strict_shared | PASS | SPICE: one_sided_95_lower=0.0028, pass=True; CHAIRs: one_sided_95_upper=0.0100, pass=True |
| coco | bepc_rescue0625 / strict_shared | PASS | SPICE: one_sided_95_lower=0.0027, pass=True; CHAIRs: one_sided_95_upper=0.0100, pass=True |
| textvqa | bepc_fast / bepc_original | PASS | accuracy: one_sided_95_lower=0.0000, pass=True |
| textvqa | bepc_rescue0625 / bepc_fast | PASS | accuracy: one_sided_95_lower=0.0000, pass=True |
| textvqa | bepc_original / strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0300, pass=False |
| textvqa | bepc_rescue0625 / strict_shared | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.0240, pass=False |

零分歧时非参数 bootstrap 可能产生退化为 0 的差值区间，无法证明未见图像上的罕见退化率为 0。尤其每任务仅 100 题，不能据此承诺总体生成质量不变。工程版还应依靠实现等价测试；定向放宽仍需独立样本及人工事实/覆盖审核。

## 与上轮七方法的历史对照

下表原样引用上轮同一批图像、长提示、512-token、单次运行的最终报告 CSV。它们属于不同运行时段，不能混入本轮配对显著性检验或据此给全部方法重新排名。本轮重新运行的 Strict 与原 BEPC 是连接两轮的控制，不复用历史速度。

| 任务 | 上轮方法 | token/s | 请求 ms | 有效 MAT | 主质量 |
|---|---|---:|---:|---:|---:|
| textvqa | ar | 28.29 | 3520.99 | NA | 70.90% |
| textvqa | vispec | 41.78 | 2360.88 | 2.01 | 71.60% |
| textvqa | strict_shared | 41.91 | 2353.50 | 2.01 | 71.60% |
| textvqa | bepc_algorithm | 43.24 | 2348.06 | 2.16 | 72.30% |
| textvqa | margin4_ccve | 42.61 | 2339.36 | 2.13 | 70.90% |
| textvqa | lvspec_paper | 68.22 | 2949.85 | 3.52 | 10.10% |
| textvqa | lvspec_adapted | 41.91 | 2353.12 | 2.01 | 71.60% |
| scienceqa | ar | 36.56 | 2995.43 | NA | 86.00% |
| scienceqa | vispec | 62.76 | 1756.50 | 2.05 | 85.00% |
| scienceqa | strict_shared | 63.02 | 1749.23 | 2.05 | 85.00% |
| scienceqa | bepc_algorithm | 64.52 | 1810.93 | 2.16 | 83.00% |
| scienceqa | margin4_ccve | 62.66 | 1752.47 | 2.10 | 86.00% |
| scienceqa | lvspec_paper | 96.45 | 2702.18 | 3.65 | 17.00% |
| scienceqa | lvspec_adapted | 63.06 | 1748.02 | 2.05 | 85.00% |
| coco | ar | 37.52 | 4197.17 | NA | SPICE 13.59% / CHAIRs 26.00% |
| coco | vispec | 67.41 | 2332.30 | 2.13 | SPICE 13.41% / CHAIRs 27.00% |
| coco | strict_shared | 67.56 | 2326.94 | 2.13 | SPICE 13.41% / CHAIRs 27.00% |
| coco | bepc_algorithm | 68.89 | 2350.17 | 2.23 | SPICE 13.57% / CHAIRs 22.00% |
| coco | margin4_ccve | 68.60 | 2340.14 | 2.23 | SPICE 13.94% / CHAIRs 21.00% |
| coco | lvspec_paper | 89.90 | 2695.08 | 3.30 | SPICE 11.58% / CHAIRs 23.00% |
| coco | lvspec_adapted | 67.61 | 2325.18 | 2.13 | SPICE 13.41% / CHAIRs 27.00% |

本轮原版与上轮原版的生成一致性：

- ScienceQA：冻结输出/路径字段一致 100/100；输入不同 0 题。若有输出差异，完整保留，不能擅自归因为 BF16、KV 或 mask。
- COCO Caption：冻结输出/路径字段一致 100/100；输入不同 0 题。若有输出差异，完整保留，不能擅自归因为 BF16、KV 或 mask。
- TextVQA：冻结输出/路径字段一致 100/100；输入不同 0 题。若有输出差异，完整保留，不能擅自归因为 BF16、KV 或 mask。

## 官方评价与复用证据

ScienceQA 使用原官方多选 Accuracy 与冻结末行答案提取；TextVQA 使用官方 VQA-style leave-one-out 归一化分数，未提取/多末行/越界答案保留并计 0。QA 评价最终答案，不审计长解释的事实或推理。

COCO 使用相同官方 PTB、CIDEr-D、SPICE 与作者 CHAIR 实现，整段描述原样输入。SPICE 仅把 Java 最大 heap 从 16G 改为 64G，防止前轮长段落 OOM；scorer、模型、公式、线程与输入均不改变。仅完整有序样本、官方标签、原始输出文字逐字一致时复用既有完整语料分数；CIDEr DF 不改变。每方法 SCORE_PROVENANCE.json 区分新计算/历史或同轮复用并封存 SHA，不复用计时。缓存写入警告与 OOM/丢句分开：后者拒绝封存评分；前者仅影响 LMDB 持久化，不改变已解析元组的内存评分。

## 局限与下一步

- 一次请求计时、L20 双卡按图像分片；没有重复运行置信度。方法在样本内随机排序减少顺序影响，仍可能受热状态和 allocator 历史影响。
- 点式 95% CI 基于 image/group 配对重采样 5000 次，所有方法共享同一组抽样；四组预先比较按 task/metric Holm 校正，不控制全部指标的全局错误率。
- 图像沿用旧实验，开发/测试分离但不是新独立测试集。没有按输出长度、正确率、是否改善或失败状态剔除样本。
- 长段落对短 caption 参考会使 CIDEr-D 极小；SPICE 受参考覆盖限制，CHAIR 仅测 COCO80 对象幻觉，不能代替属性、关系、动作、遗漏的人工评价。
- 未新增自动 Judge 或人工 Ground Truth，不由自动分数推断具体 harmful mismatch 标签。U/P 候选分量不能解释为纯视觉/语言因果效应。
- 只有当工程版路径/输出等价且实际请求时延改善时，才适合将其作为实现升级；定向放宽的采用须同时看速度、答案分数、对象幻觉和人工质量，不能只看 MAT。

原始数据、模型身份、配置、source SHA、输入、结果封印见本目录 configs/、results/ 与 audit/。主表为 BENCHMARK_SUMMARY.csv，配对差值为 PAIRED_COMPARISONS.csv，历史表为 HISTORICAL_PREVIOUS_SUMMARY.csv；图见 BENCHMARK_PERFORMANCE.pdf 与 BENCHMARK_QUALITY.pdf。

## 执行与交付核验补充

旧代码与上轮结果的137项SHA在开发阶段、主测试结束及最终交付时重新核验；均未修改。工程版与原版在全部60组开发和300组测试上的原始token、可见token、接受长度及路径完全一致。完整变化案例逐条导出为 [CHANGED_OUTPUT_PAIRS.jsonl](CHANGED_OUTPUT_PAIRS.jsonl)，包含图像引用、原始提示、两条输出与自动评分，不按好坏筛选。

CIDEr的图像bootstrap固定本次完整参考语料的document frequency和官方逐图分数；没有在每次重采样中重建DF，因此该CI是固定DF条件下的图像抽样区间。不能将其解释为重新构建参考语料的全部不确定性。

开发集CPU官方评分与主测试早期的GPU生成有短暂重叠，可能产生主机资源竞争。GPU由每个分片进程独占，方法顺序按问题随机化；这些措施仍不能消除一次计时中的全部主机/频率噪声。小幅速度差异需重复计时再确认。

统计方案中的三个主要对比及原BEPC相对Strict的背景对比全部保留；后者在读取新质量分数前通过`audit/STATISTICAL_PLAN_CLARIFICATION.json`明确纳入同一四比较Holm家族。没有据结果挑选系数。

### 预算与实际放宽幅度

| 数据集 | 方法 | 平均提交mismatch | 用满4次预算的题数/100 | 相对原版输出变化数/100 | 跳过BEPC特征的轮数/总轮数 |
|---|---|---:|---:|---:|---:|
| scienceqa | bepc_original | 3.97 | 98 | 0 | 0/3734 |
| scienceqa | bepc_fast | 3.97 | 98 | 0 | 2259/3734 |
| scienceqa | bepc_rescue0625 | 3.97 | 98 | 3 | 2251/3722 |
| coco | bepc_original | 4.00 | 100 | 0 | 0/5040 |
| coco | bepc_fast | 4.00 | 100 | 0 | 4317/5040 |
| coco | bepc_rescue0625 | 4.00 | 100 | 5 | 4338/5044 |
| textvqa | bepc_original | 4.00 | 100 | 0 | 0/3246 |
| textvqa | bepc_fast | 4.00 | 100 | 0 | 2126/3246 |
| textvqa | bepc_rescue0625 | 4.00 | 100 | 12 | 2160/3252 |

救援系数增加只改变局部条件：对同一有效救援候选，新增接受区间为 `1+0.5*v < g <= 1+0.625*v`，其中 `v=min(|m_U|,|m_P|,1)`。区间最大宽度0.125；不增加每题预算。因此分数更宽松不必然增加整题提交的mismatch总数或显著减少验证轮数。上表是已生成轨迹的描述性证据，不把预算饱和与速度变化直接解释为已识别的因果机制。
