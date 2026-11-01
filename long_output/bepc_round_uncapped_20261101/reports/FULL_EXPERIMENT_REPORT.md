# BEPC 取消每轮 mismatch 上限：完整实验报告

比较原 BEPC（每轮最多2个mismatch、每题无累计上限）与取消每轮上限后的BEPC。唯一政策变化是删除每轮mismatch计数门；候选接受评分、tau=1、双向gamma=.5、树、Prompt、EOS和512-token停止预算保持一致。无mismatch计数上限仍须逐个通过BEPC接受门，并受有限草稿树和停止规则约束。

ScienceQA、COCO、TextVQA各20开发组+100测试组，正式模型请求共600次，两臂均重新计时一次。开发样本用于实现验证，没有搜索阈值或按质量挑选方法。样本是沿用项目已有图像，不能宣称项目未见测试。

## 直接结果

- ScienceQA：取消每轮上限后token/s +0.20%，请求时延 -0.22%，有效MAT +0.41%；官方Accuracy 86.00%→85.00%。
- COCO Caption：取消每轮上限后token/s +0.13%，请求时延 -0.89%，有效MAT +0.44%；SPICE 13.09→12.91；CHAIRs 20.00%→21.00%。
- TextVQA：取消每轮上限后token/s +0.03%，请求时延 -0.72%，有效MAT +0.77%；官方Accuracy 73.30%→71.30%。

吞吐和整题时延分别评价，输出长度改变可能使二者方向不同。以下95%CI反映图像组抽样，不包含重复运行计时波动。

## 方法与执行口径

同一合法前缀固定Target候选a与Draft候选d，以原生g=z(a)−z(d)为基础。BEPC将父节点Target末层RMSNorm前状态h投影到图像prefill子空间U与正交化的近期文本子空间P，候选方向q=w(a)−w(d)，用完整h对应的RMSNorm缩放读取mU、mP。U/P是几何代理，不能解释为因果分离的视觉事实与语言偏见。mU·mP<0且两者幅度均≥1e−4时，δ=.5·sign(mP)·min(|mU|,|mP|,1)，否则δ=0；按g−δ≤1放行。residual_attenuation=false。

两臂都使用U/P秩8/4、视觉池64、文本窗口32，保留既有安全边界预筛、缓存、同一原生Target/Draft forward、tree attention与KV提交。每条合法分支连续扫描至首次拒绝，选择首个最长可接受前缀。对照在第三个mismatch前截断该分支，新方法取消这个计数截断。仅最终选中的分支提交历史与mismatch计数；其他分支的预算检查次数不是自然事件数量。没有额外Target forward。

## 冻结设置与测量

- 上游ViSpec commit：`49a33ff0ca31e281b9dc8803fd6f3041c6c0e135`。
- Target：`/data/wbb/model/Qwen2.5-VL-7B-Instruct`；Draft：`/data/wbb/model/ViSpec-Qwen2.5-VL-7B-Instruct`。
- T=0、seed=0、BF16；两张L20按物理图像组分片，同题两方法顺序随机化；单次测量，warmup排除。
- 原长输出Prompt、max_new_tokens=512、原Draft树和EOS不变；实际每条树分支仍有限长，不会无限接受。
- 请求时间包含视觉编码、prefill、Target/Draft、验证与BEPC特征；共享PIL/processor/H2D准备不计入，非客户端全链路E2E。
- token/s=可见内容token总数/同步模型请求总秒数，排除EOS，按512可见预算裁切。另列decode-only token/s。
- 有效MAT=可见范围接受Draft token总数/验证轮数，不含每轮root；原生MAT保留官方整块统计。
- 每task100个物理image/group配对bootstrap5000次；同一重采样同时用于两臂。只有一个预定方法对比/task/metric，无跨任务跨指标全局校正。

## ScienceQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 67.40 [66.16, 68.71] | 1788.13 [1709.73, 1876.76] | 77.95 | 2.42 | 2.35 | 18.56 | 19.07 |
| bepc_no_mismatch_caps | 67.54 [66.33, 68.82] | 1784.18 [1705.14, 1875.38] | 78.14 | 2.43 | 2.36 | 18.56 | 19.07 |

| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |
|---|---:|---:|---:|---:|---:|
| bepc_round2 | 86.00 [79.00, 92.00] | 0/100 | 0 | 0 |
| bepc_no_mismatch_caps | 85.00 [78.00, 92.00] | 0/100 | 0 | 0 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 每轮max | >2 mismatch轮数/总轮数 | 受影响响应数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 120.52 | 98.79 | 0.00 | 10.06 | 10.0/17.0/23.0 | 2 | 0/3629 | 0/100 |
| bepc_no_mismatch_caps | 120.50 | 98.72 | 0.00 | 10.08 | 10.0/18.0/23.0 | 3 | 6/3618 | 6/100 |

| 无上限−round2指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| accuracy | -0.0100 [-0.0300, 0.0000] | -1.16 [-3.66, 0.00] | 0.6073 |
| mean_seconds | -0.0040 [-0.0196, 0.0101] | -0.22 [-1.11, 0.56] | 0.6043 |
| throughput_tokens_per_second | 0.1381 [-0.0680, 0.3744] | 0.20 [-0.10, 0.56] | 0.2048 |
| answer_extraction_failure_fraction | 0.0000 [0.0000, 0.0000] | NA [NA, NA] | 1.0000 |
| effective_mean_accepted_draft_length | 0.0096 [-0.0002, 0.0227] | 0.41 [-9.222e-03, 0.97] | 0.0994 |

全部转移（round2→无mismatch上限）：`{"acceptance_path_changed": 6, "acceptance_path_unchanged": 94, "outputs_changed": 6, "outputs_unchanged": 94, "round2_0_to_uncapped_0": 14, "round2_1_to_uncapped_0": 1, "round2_1_to_uncapped_1": 85, "score_degraded": 1, "score_tie": 99}`。

原round2在62/100个响应中记录计数门触发，候选扫描访问累计180次；这些访问不是去重事件、选中轮次或独立mismatch。
## COCO Caption

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 75.07 [74.23, 75.90] | 2305.98 [2223.92, 2391.57] | 83.81 | 2.71 | 2.67 | 18.31 | 19.07 |
| bepc_no_mismatch_caps | 75.17 [74.36, 75.97] | 2285.52 [2212.84, 2357.35] | 84.00 | 2.72 | 2.69 | 18.31 | 19.07 |

| 方法 | CIDEr-D原尺度 | SPICE % [95% CI] | CHAIRs % [95% CI] | CHAIRi % |
|---|---:|---:|---:|---:|
| bepc_round2 | 2.396e-39 | 13.09 [12.26, 13.97] | 20.00 [12.00, 28.00] | 5.37 |
| bepc_no_mismatch_caps | 2.396e-39 | 12.91 [12.05, 13.80] | 21.00 [13.00, 29.00] | 5.89 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 每轮max | >2 mismatch轮数/总轮数 | 受影响响应数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 173.12 | 145.77 | 0.00 | 27.34 | 27.0/37.1/60.0 | 2 | 0/4738 | 0/100 |
| bepc_no_mismatch_caps | 171.80 | 144.52 | 0.00 | 28.26 | 29.0/39.0/45.0 | 4 | 54/4687 | 40/100 |

| 无上限−round2指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| SPICE | -0.0018 [-0.0048, 0.0008] | -1.39 [-3.66, 0.59] | 0.1902 |
| CHAIRs | 0.0100 [0.0000, 0.0300] | 5.00 [0.00, 18.75] | 0.6303 |
| mean_seconds | -0.0205 [-0.0626, 0.0173] | -0.89 [-2.67, 0.76] | 0.3295 |
| throughput_tokens_per_second | 0.0947 [-0.2532, 0.4555] | 0.13 [-0.34, 0.61] | 0.5997 |
| answer_extraction_failure_fraction | NA [NA, NA] | NA [NA, NA] | NA |
| effective_mean_accepted_draft_length | 0.0118 [-0.0063, 0.0306] | 0.44 [-0.23, 1.14] | 0.2078 |

全部转移（round2→无mismatch上限）：`{"CHAIRi_equal": 97, "CHAIRi_increased": 3, "CHAIRs_equal": 99, "CHAIRs_increased": 1, "CIDEr_decreased": 19, "CIDEr_equal": 60, "CIDEr_increased": 21, "SPICE_decreased": 22, "SPICE_equal": 60, "SPICE_increased": 18, "acceptance_path_changed": 40, "acceptance_path_unchanged": 60, "outputs_changed": 40, "outputs_unchanged": 60}`。

原round2在95/100个响应中记录计数门触发，候选扫描访问累计589次；这些访问不是去重事件、选中轮次或独立mismatch。
## TextVQA

| 方法 | token/s [95% CI] | 请求ms [95% CI] | decode token/s | 原生MAT | 有效MAT | 峰值allocated GiB | 峰值reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 46.66 [45.40, 47.90] | 2455.56 [2366.94, 2549.92] | 75.46 | 2.45 | 2.39 | 22.56 | 34.63 |
| bepc_no_mismatch_caps | 46.67 [45.35, 47.99] | 2437.79 [2345.79, 2533.82] | 75.81 | 2.47 | 2.40 | 22.56 | 34.63 |

| 方法 | 官方Accuracy % [95% CI] | 提取失败总数 | invalid | uncertain |
|---|---:|---:|---:|---:|---:|
| bepc_round2 | 73.30 [65.00, 81.00] | 3/100 | 3 | 0 |
| bepc_no_mismatch_caps | 71.30 [63.10, 79.00] | 4/100 | 4 | 0 |

| 方法 | token/题 | 词/题 | 截断% | mismatch均值 | p50/p95/max | 每轮max | >2 mismatch轮数/总轮数 | 受影响响应数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bepc_round2 | 114.57 | 91.47 | 0.00 | 13.90 | 14.0/23.0/30.0 | 2 | 0/3414 | 0/100 |
| bepc_no_mismatch_caps | 113.77 | 90.59 | 0.00 | 14.46 | 14.5/24.0/30.0 | 3 | 35/3372 | 30/100 |

| 无上限−round2指标 | 差值 [95% CI] | 相对变化% [95% CI] | 近似双侧p |
|---|---:|---:|---:|
| accuracy | -0.0200 [-0.0500, 0.0000] | -2.73 [-6.95, 0.00] | 0.2757 |
| mean_seconds | -0.0178 [-0.0499, 0.0143] | -0.72 [-2.03, 0.59] | 0.2777 |
| throughput_tokens_per_second | 0.0120 [-0.4084, 0.4540] | 0.03 [-0.87, 0.97] | 0.9576 |
| answer_extraction_failure_fraction | 0.0100 [0.0000, 0.0300] | 33.33 [NA, NA] | 0.6369 |
| effective_mean_accepted_draft_length | 0.0184 [-0.0027, 0.0429] | 0.77 [-0.11, 1.81] | 0.1074 |

全部转移（round2→无mismatch上限）：`{"acceptance_path_changed": 30, "acceptance_path_unchanged": 70, "outputs_changed": 30, "outputs_unchanged": 70, "score_degraded": 2, "score_tie": 98}`。

原round2在85/100个响应中记录计数门触发，候选扫描访问累计358次；这些访问不是去重事件、选中轮次或独立mismatch。

## 自动代理非劣效检查

沿用预先固定容忍门：QA最多下降2个百分点；COCO同时满足新SPICE≥原SPICE的95%、CHAIRs增幅≤2个百分点。使用配对bootstrap单侧95%界；比较边界预先规定1e−12纯算术舍入容差，原始未舍入界保留在JSON。这不是额外质量容忍放宽，不作用于CIDEr或配对差异；不得将“不显著”视为相等。这些容忍门是pilot研究扩展，不是官方指标规定的安全标准。

| 任务 | 自动代理门 | 原始单侧界 |
|---|---|---|
| scienceqa | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.030000000000000027, pass=False |
| coco | NOT ESTABLISHED | SPICE: one_sided_95_lower=0.002313755979957839, pass=True; CHAIRs: one_sided_95_upper=0.03, pass=False |
| textvqa | NOT ESTABLISHED | accuracy: one_sided_95_lower=-0.04999999999999982, pass=False |

QA仅评价最终答案，不验证全部解释；COCO双人独立人工事实/充分性审核仍pending。原样保留invalid、uncertain、截断与提取失败，失败计0，不删除差例。零分歧的bootstrap可能退化，不能推出罕见风险为0。

## 工程成本与复现审计

FEATURE_COST_COUNTERS.csv保留候选、子空间、缓存等实际计数覆盖。它们不是单独同步GPU延迟；请求总时间与显存已包含真实成本。allocated为活跃张量峰值；reserved受allocator历史影响，不能解释为算法最低独占显存。两臂均无额外Target forward。

- ScienceQA round2对照与上一轮BEPC：完整输出/接受路径精确一致100/100，输入差异0；未复用历史时延。
- COCO Caption round2对照与上一轮BEPC：完整输出/接受路径精确一致100/100，输入差异0；未复用历史时延。
- TextVQA round2对照与上一轮BEPC：完整输出/接受路径精确一致100/100，输入差异0；未复用历史时延。

audit/MAIN_VALIDATION.json与DEVELOPMENT_VALIDATION.json审计运行期预算字段、提交路径和首次分叉。STATISTICAL_ANALYSIS.json另保留所有首次选中路径分叉，不按质量挑选。CHANGED_OUTPUT_PAIRS.jsonl包含全部改变完整输出、输入图像引用、Prompt及官方质量分数。

CPU官方评分仅在全部GPU生成退出、GPU_COMPLETION.json确认无GPU PID后开始。整语料官方分数仅在完整有序样本、标签、解析器、实现和原始输出签名完全相同时复用；记录来源与SHA，不复用计时。CIDEr-D的文档频率语料不变，bootstrap使用固定官方每图分数，不在每次抽样重拟合DF。

## 评价局限

- ScienceQA使用冻结末行解析器和官方Accuracy；TextVQA执行官方VQA-style归一化与leave-one-out评分。全部提取失败计0。
- COCO全文原样进入官方PTB、CIDEr-D、SPICE和作者CHAIR；SPICE仅使用既有64G Java资源上限，评分模型/公式不变。
- 长段落对短参考使CIDEr-D可能接近0，不得单独解释为事实差；SPICE受参考覆盖限制；CHAIR主要覆盖COCO对象幻觉，无法验证全部属性、动作、关系、数量与遗漏。
- 本轮只有一次运行，复用项目已有图像；CI不涵盖重复运行噪声、设备间推广或罕见错误，没有独立未见图像与人工caption质量证据。
- 政策变化会改变后续上下文与输出长度；这是完整解码策略对比，不是单事件mismatch的因果有害标签。
- 未调整BEPC算法思想、阈值或候选信号；没有训练、融合、扩样或根据测试成绩挑选更优参数。
- 若自动代理非劣效未建立，应保留原方法，不能用吞吐或MAT增长替代质量证据。

证据：configs/frozen_config.json、results/main/、metrics/main/与audit/STATISTICAL_ANALYSIS.json。数值见BENCHMARK_SUMMARY.csv、PAIRED_COMPARISONS.csv；图见BENCHMARK_PERFORMANCE.pdf、BENCHMARK_QUALITY.pdf。

## 事后描述性机制检查：多接受 token 与同长度换分支

本节读取全部600条主测试原生日志，不使用质量标签筛选案例，也不修改评分或接受规则。在两臂共同历史仍完全一致的首次路径分叉处，分别比较选中前缀长度与树行号。

| 任务 | 实际>2 mismatch轮数/总轮数 | 影响题数/100 | 首次分叉长度增加 | 同长度换更早分支 |
|---|---:|---:|---:|---:|
| scienceqa | 6/3618 (0.17%) | 6 | 6 | 0 |
| coco | 54/4687 (1.15%) | 40 | 25 | 15 |
| textvqa | 35/3372 (1.04%) | 30 | 21 | 9 |

同长度换分支的原因是保留的first-longest规则：取消次数门后，原来较早的某条分支能够达到已有最佳长度，于是取代原来较晚的分支；当前轮提交token数量相同，但文本和后续上下文发生变化。因此不能把所有输出变化都解释成多接受了token。只有少量轮次实际越过原次数门，这提供了本轮MAT和吞吐收益较小的结构性解释；整段时延还受后续轨迹、输出长度与特征开销影响，不能由该计数单独分离。

全部首次分叉与图像/题号见`audit/ROUND_LIMIT_EFFECT.json`。冻结的官方评分、统计清单、CSV和曲线均未修改。本补充章节的前后哈希见`audit/REPORT_ADDENDUM_PROVENANCE.json`；最终交付以`audit/FINAL_DELIVERY.json`为准。`audit/REPORT_DELIVERY.json`保留添加本章节前的生成器封印。
