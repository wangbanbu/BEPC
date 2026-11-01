# 同一原生验证树上的前缀诊断

本次补充核查不生成新的完整 Caption、不变更算法或标签。先使用历史 BEPC/Margin 的选路日志定位首次路径分叉轮，在原模型、原处理器、原配置和原 BEPC 控制器上重放到该轮，生产决策完成后停止，尚未提交该轮 KV 或继续生成。

`source/prefix_tree_diagnostic.py` 为最初 6 + 43 例入口；后续 `source/prefix_tree_diagnostic_all_changed.py` 是单独复制的扩展入口，仅增补全部可见 token 序列发生变化的样本，不更改最初入口或其已保存输出。两者直接只读导入父实验封存 harness，不调用会写入父实验的 Runtime 初始化。

每例严格验证图像文件 SHA、完整 processor 输入张量 SHA、Prompt token IDs、image_grid_thw。首次路径分叉前的每轮 BEPC 路径须与两方法历史路径完全相同；该轮 BEPC 路径须与历史 BEPC 完全一致。

同一棵树上，使用原 `budgeted_scan` 执行 shadow Margin，保持 EOS、可见长度边界、每轮 2 mismatch、无整题上限及原行序 first-longest tie 规则，结果须与历史 Margin 分叉轮路径完全一致。只有这些核对全部通过才记 valid。

先保存实际生产 BEPC 访问过的门、缓存决策、pruning 原因及已选路径，再对生产 BEPC 或 shadow Margin 访问过的候选补充冻结数学公式的完整分解。补充读出的 full score 若与原生产 BEPC 缓存决策不同，直接标 invalid。分解使用原当前树的 parent hidden，以及请求内已建立的 U 和合法历史 P，无串行重新 forward 替代树状态。

每例 JSON 保存完整树候选、retrieve、Target top1、native Margin、shadow 扫描行结果以及候选级 g/m_U/m_P/m_R/delta/score、实际访问与选中边标志。`production_bepc_visited` 与 `shadow_margin_visited` 分别保存，不能把 shadow 或未选候选当作实际提交事件。

每次核查本身重新执行必要的 image prefill 和先前 Target/Draft tree forwards，`model_calls` 完整计数。`extra_diagnostic_target_forwards=0` 仅表示在已有该轮验证状态上提取分量没有新增 Target forward；不表示整次核查没有计算成本。`elapsed_seconds` 是诊断重放时间，包含附加读出与输入处理，不可替代正式生产速度测量。

原生 forward、tree mask、mRoPE 与 KV 更新使用封存实现；所有先前路径和同树 shadow 结果均做逐例核对。本入口未额外保存原生 mask/position tensor 快照，不能将其描述为独立串行 mask 数值对照实验。

# 分叉归因

`source/summarize_prefix_diagnostics.py` 仅做 CPU token ID 对齐。它把候选门位置与两条可见输出的首次 token 差异位置分别保存，且检查候选合法前缀哈希是否等于观测共同前缀。

- `direct_protection_Target_retained`：同位置、同合法前缀下，Margin 接受并提交 d，BEPC 拒绝 d，实际提交 a，即当时 Target greedy token。这里的 a 是共同前缀下局部 Target 选择，不保证等于另一条完整 Strict rollout 在不同上下文下的 token，更不自动保证事实正确。
- `direct_protection_alternative_token`：同样阻止 d，但 BEPC 实际换成第三个 token，不能称保住原 Target token。
- `direct_rescue_Draft_committed`：同位置下，BEPC 额外放行 d，Margin 实际生成 a。
- `direct_rescue_alternative_comparison`：BEPC 放行 d，Margin 最终在该位置为第三个 token。
- `descendant_gate_changes_earlier_ancestor_selection`：深层门改变最长路径或 tie 的胜出行，导致更早的祖先 token 改变。不能把可见祖先变化错误地说成对该祖先直接拒绝。
- 其他类别保留不同块边界、后续才发生 token 差异和未直接归因的情况，不强制归入 protection/rescue。

上述均为机械决策/选路证据，不是有害标签或事实正确性结论。最终答案得分提高或 CHAIR 改善属于整条历史策略轨迹的结果，未经单事件对照不证明最初那个门造成全部改善。

# 实际访问与 hypothetical gate 的严格分离

完整候选 JSON 有意保留 `union(实际 BEPC 访问, shadow Margin 访问)` 的读出。其中 7 条分歧候选仅由 shadow Margin 访问，其 BEPC 分解是该树上的假设读出，不能称实际发生的 protection/rescue。最初 `allchanged/` 汇总作为分析过程快照保留；用于正式报告的严格汇总是 **`allchanged_actual_visited_v2/`**，只把 `production_bepc_visited=true` 的分歧门用于机制归因。

这次修正不改变 126 例直接保留 Target token 或任何首次可见差异归因数量，但改变部分样本“第一棵分叉树含 protection/rescue 两种门”的类别。原始候选读出和旧汇总均未覆盖。实际访问也是机制归因的必要条件，仍不替代单门干预或图像事实真值。

`DIAGNOSTIC_VALIDATION.json` 对 276 例执行 31,996 项 CPU 检查，包含 frozen score 公式、位置与候选对齐、历史选路、额外读出与生产门一致，以及父实验结果/源码 SHA 不变，全部通过。
