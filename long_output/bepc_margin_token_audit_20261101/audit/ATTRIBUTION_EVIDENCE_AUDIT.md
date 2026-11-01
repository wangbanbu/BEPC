# BEPC–Margin token protection 的证据边界审计

本文件仅审计历史日志与实际控制器，未运行模型、未更改历史源码、配置或结果。后续新增的共同树诊断及其结论应单独留存，不覆盖本历史证据快照。

## 1. 历史数据究竟保存了什么

主实验共有 600 条请求记录：ScienceQA、COCO、TextVQA 各 100 题 × 两种策略。全部 600 条均为 `diagnostic_recording=false`、`rich_diagnostics=false`。三个 `scored_conflict/protection/rescue_candidates` 字段全部为 null，而非零。

保存了原始/可见 token IDs、完整输入标识、最终文本、官方任务分数，以及每轮选中路径的 `round,row,accepted_draft,mismatches,response_mismatches_before,nodes,tokens`。未保存整棵候选树、每个候选的 Target top1、native Margin、BEPC m_U/m_P/delta。因此可重建 token 编辑、实际选中块和最终任务结果，不能仅从旧路径日志宣称某个 token 由 protection 而非 rescue 改变。

证据源码：`harness/speed_controller.py:25` 关闭生产丰富诊断；`:142` gate 与缓存；`:187` 扫描和选路；`:213` 仅保存选中路径；`:242` 将未采集候选诊断显式设 null。`harness/refined_controller.py:41` 定义真实 EOS、长度、独立分支预算和连续前缀扫描。`harness/run_benchmark.py:242` 后才进行 EOS/输出预算裁剪，原始块可能包含不可见后缀。

## 2. 两个不同的“首次分叉”

1. 首次路径/执行分叉：相同已提交前缀之后，行号、节点、接受长度或所提交块首次不同。
2. 首次可见 token 分叉：两条最终可见 token 序列的最长共同前缀之后首次不同。

二者不能合并。同一棵树换行可能仍选择完全相同的 token 前缀；一个方法提交较长块，另一个下一轮以 root 提交相同 token，也会先出现块边界变化、后出现可见分叉，甚至全文相同。行号或树节点编号只在同一轮同一棵树内有意义，不可跨不同前缀/不同轮比较。

原始每轮选中 `tokens` 的拼接应先与 `raw_token_ids` 完全核对；可见报告以 EOS 和 512-token 预算裁剪后的区间为准。首次块分叉后不能继续按相同 round index 假定共同上下文。

## 3. 最小而可靠的机制诊断

在新代码目录使用原 BEPC 入口重放至共同的首次路径分叉树，保持原 gate、pruning、U/P 构建与缓存、原始 forward 和停止规则。先运行生产决策并保存实际缓存，再做观测性附加读出，避免附加完整评分改变惰性初始化或后续扫描状态。

在这一棵原生验证树上捕获全部候选/retrieve、Target top1、原生 FP32 差分 Margin、每条路径预算扫描结果。另以完全相同的 `budgeted_scan`、EOS/边界/round2/无整题上限、原行序 first-longest tie 规则执行 shadow Margin。shadow Margin 选路应与历史 Margin 在首次分叉树的选择匹配，BEPC 则应与历史 BEPC 匹配；两者任何不匹配都需记 unresolved。

对真正 gate 分歧候选记录 `(parent,a,d)`、位置与合法共同前缀、g、m_U、m_P、m_R、delta、score、gate、available、两种接受决策、原实际访问/是否影响 winning row。父状态是预测 d 的 parent，不能使用 d 本身或未来节点 hidden。U/P 必须来自相同请求 image/prompt 与合法历史。候选不在实际扫描可达路径上，必须标记 off-path，不算实际被保护的 committed token。

直接把完整文本 prefix 串行重新 forward 只能作为额外诊断：BF16 GEMM 形状、分块 KV/mRoPE、原始树中隐藏状态及历史状态来源均可能不同。未验证历史 token/path 与相关门决策一致时，不可把串行读出的分量当作原门的精确历史记录。

## 4. 归因类别

- **决策 protection**：同一 parent/candidate 上普通 Margin 接受 d，而 BEPC 拒绝 d。该分类描述决策方向，不代表 d 已经被证实有害。
- **决策 rescue**：普通 Margin 拒绝 d，而 BEPC 接受 d。同样不预设任务收益。
- **直接 Strict token 保护**：还必须显示共同前缀首次可见差异处，Margin 提交 d，BEPC 提交该前缀原生 Target top1 a，且相应 protection gate 确实影响选路。
- **替代候选选路**：BEPC 拦下 d 后提交其他非 a 候选，只能称候选阻止/换路，不可称原样保住 Strict token。
- **相同文本、不同块边界**：没有当前可见 token 替换；后续可能受执行轨迹改变影响。
- **下游答案/事实变化**：最终答案改善、恶化是整条策略轨迹的配对结果，不自动等于首次 token 已经因果保护关键事实。
- **Unresolved**：重放不一致、无共同前缀、缺诊断、数值边界不稳等，全部保留。

## 5. “关键 token”的审查边界

ScienceQA 可以客观标记固定答案解析器定位到的最终选项 span；TextVQA 使用官方归一化及分数，区分终端答案文本变化、格式提取失败和真实得分改变。若答案改变发生在最初 gate 分歧很久以后，报告“下游答案改变”，不伪称那个答案 token 直接获得 protection。

COCO 对象/属性/数量/动作/关系需要对齐完整词或原子事实，BPE token 变化不是完整语义变化；语言重述也不是事实保护。自动 CHAIR 错误名词减少是辅助候选证据，不能替代看图事实审查。SPICE/CIDEr 上升不能直接标记“关键视觉 token 得到保护”。

要证明单个决策的任务因果收益，另需同一前缀只干预该 gate、随后固定恢复同一策略的单事件反事实。本轮原有两条完整 policy rollout 只支持策略级差异与机制关联；不新增该因果实验也可以完成当前 token 审计，但应明确证据等级。

## 6. 审稿结论

旧日志足够完整完成 token 编辑与官方任务分数审计；不足以直接数出真实 protection/rescue，更不足以把整条输出收益都归因于首个 mismatch。补充同一原生首次分叉树的 shadow gate 诊断后，可以验证候选被阻止/救援及是否直接保留 Target token。关键事实正确性仍需独立图像审查；单决策任务因果结论仍需额外单事件实验。
