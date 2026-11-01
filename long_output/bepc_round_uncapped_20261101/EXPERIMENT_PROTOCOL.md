# BEPC：取消每轮 mismatch 次数限制的直接解码对照

用户要求在已取消整题累计 mismatch 上限的基础上，进一步取消每轮至多2次限制，测试速度与质量。原代码、历史结果和冻结评分保持不变。本轮在独立目录执行固定规则消融。

两臂：

- `bepc_round2`：上一轮 BEPC，每轮最多2次 mismatch，整题无总上限。
- `bepc_no_mismatch_caps`：同一 BEPC，取消每轮和整题的次数上限。每个 mismatch 仍必须通过原有评分与合法性检查。

不修改 BEPC 分数：原生 Target top1与Draft候选margin `g`，U8/P4、视觉池64、文本窗口32、gamma_rescue=gamma_protection=.5、kappa=1、eta=1e-4、tau=1、双向冲突门、residual_attenuation=false、相同数值预筛和缓存。初始化时以旧round2通过只读父类检查，任何forward/决策前将实例round_budget改为正无穷；JSON配置与统计保存null，不以巨大有限数替代无上限。

共同采用 Qwen2.5-VL-7B + 原 ViSpec Draft、T=0/seed0/BF16、相同长输出任务Prompt、512-token预算、原生树与KV/mRoPE/causal mask/EOS及停止逻辑、连续前缀扫描和首个最长分支、仅选中路径提交。真实树为29个Draft节点加根；depth3实现先展开1层再循环3次，因此最长路径最多4个Draft后代。取消次数限制不增加树深度、不忽略EOS/可见边界，也不是无条件接受候选。

ScienceQA、COCO、TextVQA 各20开发组、100测试组，沿用同一冻结样本、原始图像和输入。开发集只检查数值和执行完整性，不调阈值；主集每方法运行一次，共600请求。两张L20按image/group分片，同题两方法固定随机顺序，重新计时两臂，不混用历史时延。加载和warmup排除、输入预处理复用。全部失败、提取失败、截断、EOS与原始输出保留。

操纵核验：控制臂全部原始/可见token、接受长度、选中节点和输入须精确复现上一轮；两臂同题输入一致。取消约束臂的首次选中路径分叉必须依赖第三个或更多 mismatch 可被接受：共同历史一致时，不得在所选路径仍≤2次时出现首次分叉。统计每轮实际>2的轮数、题数、最大值，区分所有分支预算binding计数和实际选中路径；不将off-path扫描次数当实际接受。

质量实现：沿用官方ScienceQA Accuracy、TextVQA VQA-style Accuracy、固定末行答案解析器（失败计0），COCO完整长Caption用官方CIDEr-D、SPICE和作者CHAIR。整语料评分复用仅限样本、顺序、标签、解析器、官方实现与全部原始文本一致；保留复用封印。SPICE继续只使用64G Java资源上限。官方评分在GPU推理全部结束后执行。

速度同时报告内容token/s、仅decode token/s、同步请求时延、输出长度、MAT（接受Draft token/验证轮数，根不计）、allocated/reserved峰值。请求时间含模型prefill/视觉编码、解码、验证及BEPC成本，不含已共享的图像处理/H2D，不称客户端E2E。

预定统计：按物理image/group配对bootstrap5000次，95%区间；唯一主对比为 `bepc_no_mismatch_caps − bepc_round2`。单次计时CI仅反映图像抽样，不含重复计时波动。QA最多下降2pp；COCO SPICE至少对照.95且CHAIRs增加至多.02，使用单侧95%界的自动非劣诊断。浮点门固定使用1e-12算术舍入容差，保留原始界，避免恰等于阈值时误判；该容差不是质量阈值调整。

质量分数和容忍规则冻结，不根据AUROC或结果调参。QA不核验整段解释事实；长段落CIDEr存在长度趋零问题；SPICE受参考覆盖限制，CHAIR主要覆盖对象。COCO双人图像盲评仍pending，自动分数不称人工Ground Truth。当前图像为项目已用样本，不作项目未见独立测试或一般性质量保持结论。
