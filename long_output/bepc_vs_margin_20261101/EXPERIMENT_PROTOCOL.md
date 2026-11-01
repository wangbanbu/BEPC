# 无总次数上限 BEPC 与简单 Margin：生成速度与质量对照

本轮固定比较当前 BEPC 与普通 Margin，不修改历史 BEPC 代码及已有测量。

两种方法共用 Target Qwen2.5-VL-7B、ViSpec Draft checkpoint、原生验证树（29 Draft 节点加根）、T=0、seed=0、相同图像和长输出 Prompt、512 token 输出预算、原生 KV/mRoPE/tree mask/EOS/停止规则。沿用 ScienceQA、COCO、TextVQA 各 20 开发组和 100 测试组；主测试各方法执行一次，共 600 请求。开发组只做完整性核验，不选择阈值。

- `margin_no_response_cap`：有限原生 Margin `g = FP32(z_top1) - FP32(z_draft) <= 1`。
- `bepc_no_response_cap`：完整沿用上一轮 U8/P4、文本窗口 32、视觉池 64、gamma=.5、kappa=1、eta=1e-4、tau=1、无 residual attenuation 的双向有界修正；在合法分量冲突时 `delta=.5 sign(m_P) min(|m_U|,|m_P|,1)`，接受 `g-delta<=1`。

共同约束：每轮最多 2 个 mismatch；整题无累计次数上限；相同连续前缀扫描、首个最长分支、仅选中路径提交与计数。Margin 不安装视觉、文本或 Draft 概率观察器。两种方法同用 compact logits 和 terminal CPU selection hints。比较仅移除 BEPC 修正及其必要计算；不进行接受率匹配或阈值扫描。

两张 L20 按物理图像组独立分片。每题方法顺序固定种子随机化，输入预处理复用；每方法独立重置 KV、seed 和峰值计数。加载及每方法 warmup 不计入测量。重新测量两种方法，历史计时不混入主表。BEPC 的输入、原始/可见 token 和选中树路径须逐条匹配上一轮。保留全部失败、EOS、截断、答案提取失败与原始文本。

速度主指标为可见生成 token / 同步请求耗时，另列仅解码 token/s、每题时延、输出长度、MAT（接受 Draft token/验证轮数，根不计）和峰值 allocated/reserved 显存。完整质量评分在两个 GPU 进程退出后执行：ScienceQA 官方 Accuracy；TextVQA 官方 VQA Accuracy；COCO 官方 CIDEr-D、SPICE 与 CHAIR，完整长 Caption 原样评分。答案解析失败保留且计 0 分。只可复用有封印、样本顺序与整套原始输出完全一致的评分；不得复用历史计时。

统计预先固定为按物理 image/group 配对 bootstrap 5000 次、95% CI，每任务/指标唯一主对比 BEPC−Margin。单次主测试的 CI 衡量图像抽样差异，不覆盖重复计时波动；既有测试图像也不作为项目未见独立测试。自动质量非劣诊断沿用 QA 最多下降 2pp；COCO SPICE 至少基线 .95 且 CHAIRs 至多增加 .02，要求单侧 95% 配对 bootstrap 同时满足。该诊断不等于人工事实质量保证。

COCO 双人图像盲评仍 pending。长段落与短参考会造成 CIDEr 长度惩罚和数值趋近零；不能由 CIDEr 或单个自动指标推导 factual harmful。QA Accuracy 仅评价最终答案，不证明解释事实正确。所有改变输出的成对案例均导出，避免选择性报告。结果不足以证明相同质量下更快时，明确保留该限制。
