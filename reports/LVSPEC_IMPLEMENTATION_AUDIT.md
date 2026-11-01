# LVSpec 在干净 ViSpec 上的实现审计

日期：2026-10-31。状态：论文、官方源码和 CPU 数值/控制器测试已核对；本文不预先声称真实模型 GPU parity、任务质量或加速结果。实现只写入本新 benchmark 根目录；未读取或移植旧实验的绝对 V1 阈值。

## 来源与身份

- 正式论文：[ACL Anthology](https://aclanthology.org/2026.acl-long.1087/)，[正式 PDF](https://aclanthology.org/2026.acl-long.1087.pdf)。重点核对 §3.1–3.3、Implementation Details 与 Appendix E/I。
- 同作者版本：[arXiv HTML](https://arxiv.org/html/2604.05650)。论文参数与正式 PDF 一致。
- fresh clone：[官方 GitHub](https://github.com/zju-jiyicheng/LVSpec)，本地 `reference/LVSpec/`，commit `84dde16ea16b39402552eb4eb95d26f9c5101d15`，提交时间 `2026-07-04T12:44:01+08:00`。检索时 `git status --short` 为空。没有调用旧根中的 LVSpec 副本。
- 新生成代码入口：`harness/lvspec_controller.py:LVSpecController`，接口与本 benchmark runner 的 runtime/model/variant/prompt/grid 约定一致。

| 官方文件 | 本次核对内容 | SHA256 |
|---|---|---|
| `decoding/loose_match.py` | 低视觉相对排名与 PST 的实际接受逻辑 | `e899f8d346024409e40b427c941b587cdada44c938f845b8084e1b0883f41bdc` |
| `decoding/tree_decoding_qwen2_5.py` | Top-N cosine、parent/logit 对齐、prefill reference、接受分支 | `a8ecb27fc1c7c43b9bcadb1b8560e6df5c623ba8153933e48a66049b6f6b14e5` |
| `inference.py` | quick-run 的 LVSpec 与 PST 开关 | `607f48306606d2a1dcf5f9dd761be45586ae144eb2c7f87485608bdcd36c00ff` |
| `evaluation/qwen2_5_vl_lvspec.py` | lmms-eval 实际选择 PST 路径 | `55560d75eb9183a18bd554a2552db0bd5df571b1a13da33d2e97190c707d00cf` |
| `models/modeling_qwen2_5_vl.py` | 最终 norm 后、lm_head 前状态与输出字段 | `c2d30524377c785d437e3c983c6624bdecfb32d2ef4a04b3d1b2c2890cf19457` |

## 论文规则与源码路径

论文用最后层文本/视觉表示的 Top-N cosine 均值衡量视觉相关性，在每个 draft step 放松最低相关性的比例 λ；PST 允许 Target 选择在当前 draft span 另一位置出现。正文默认 λ=.7、N=10、chain K=10。Appendix E 明确沿每个合法 branch 验证、再取最长接受 branch，支持本实现的分支局部比较集合。[正式 PDF](https://aclanthology.org/2026.acl-long.1087.pdf)

相应公开代码为 [visual similarity / evaluate_posterior](https://github.com/zju-jiyicheng/LVSpec/blob/84dde16ea16b39402552eb4eb95d26f9c5101d15/decoding/tree_decoding_qwen2_5.py#L492) 及 [loose_match_hidden_topk_or_search](https://github.com/zju-jiyicheng/LVSpec/blob/84dde16ea16b39402552eb4eb95d26f9c5101d15/decoding/loose_match.py#L239)。实际完整生成路径为 `inference.py`/lmms wrapper → `SD_generate_with_pruning` → native tree forward → `evaluate_posterior` → helper → 原生 update。

本移植保留的数学为：

\[
A(n)=\operatorname{mean}\operatorname{TopN}_{v\in V}
 \cos(h_n^{\mathrm{finalnorm}},h_v^{\mathrm{finalnorm}}),\quad N=10.
\]

令一个真实 retrieve 行为 `(n_0,…,n_L)`、candidate 为 `(c_0,…,c_L)`。`logits[row,j]` 预测 `c_(j+1)`；公开代码用 `tree_hidden_states[:, :-1]`，因此本实现的相对分数是父节点 `A(n_j)`。**它不是已消费候选子节点 `A(n_(j+1))`；不是将兄弟候选错误绑定到同一隐藏状态，而是显式保留公开代码的 parent-state 定义。**

对该行前 `W=min(L,10)` 个验证位置排序，放松最低 `floor(λW)` 个位置。该位置若 mismatch，可因低相对视觉排名或同 branch 的 PST 成功而接受。身份相同仍优先接受。整行采用连续前缀扫描，并按 first-longest 选择 branch。没有概率、entropy、margin 的附加门。

## 必要适配及论文—代码差别

| 问题 | 论文/公开实现 | 本次决定及归属 |
|---|---|---|
| 原始形态 | 主要 chain K10；Appendix E 两 branch 各 K10 | 采用当前 ViSpec 原树，每条真实 root→leaf 行分别排序；不更改 Draft 树。属 ViSpec 适配，不能宣称原 K10 chain 重现实验。 |
| 相对松弛数量 | 论文 λ=.7；公开 helper 绝对 `k=7` 且 clamp 到至少1 | 映射为 `floor(λ*min(actual_branch_positions,10))`。短树不能直接 clamp 绝对7，否则常将所有位置放松。λ=0 允许零视觉放松。属明确比例适配。 |
| sibling/local set | 公开 helper 每行独立排序 | 不将不同分支/不同祖先路径节点混成全树 pool。共享 parent 的 absolute 分数相同，但在不同合法行的 rank 可以不同；这是分支局部上下文策略，不是候选固有分数。 |
| H_V 参考范围 | 论文定义真实视觉 token 状态；公开 one-stage Qwen 路径把整段 prefill 的 `output.hidden_states` 传入 cosine，未筛视觉行 | 本主实验按 `image_token_id` 取真实视觉行，明确采用论文定义；不把公开代码混入 system/question 行的行为伪称纯视觉参考。不新增 all-prefill 对照臂。 |
| 状态层次 | 原 Qwen `outputs[0]` 为 final model.norm 后、lm_head 前 | 直接 hook 新 Target `model.norm` 输出，图像和 parent 坐标完全一致；不沿用 BEPC pre-norm block 输出。 |
| PST | 正文是 draft span；公开 helper 搜索整个 candidate 行，包括 root seed | 默认包含 root，限定当前真实 branch 的最多 root+10 Draft token；不搜索 sibling。显式保留公开 helper 的 root 细节。 |
| PST 开关 | quick-run `lvspec` 不含 PST；`lvspec_with_pst` 包含；lmms 的 `lvspec` 实际包含 PST | 两个主 LVSpec 比例臂均开启 PST，作为完整方法；不会悄悄把 quick-run 无 PST 版本当成完整方法。 |
| λ0 | w/o PST 可退到 exact；有 PST 时仍可接受 mismatch | λ0 是网格成员，**不是 Strict fallback**。若预定开发质量门全部失败，必须另设 `disable_relaxation=True`，并把臂标注为 Strict fallback。 |
| 熵/概率 | `evaluate_posterior` 计算 entropy，但 hidden_topk/PST 接受分支没有读取它 | 不复制无作用的全词表 softmax/entropy 开销，也不发明概率门。省去 dead computation 不改变本规则。 |
| ties | 公开 argsort 未声明 stable | 明确相同分数按当前分支位置稳定排序，是可复现的 tie 约定。 |
| mismatch 预算 | LVSpec 原方法没有每轮2/整响应4限制 | 原样保留无限 mismatch 预算；不能和本项目 BEPC/ME 2/4 预算混称完全匹配。预算差异必须在主表/讨论披露。 |
| EOS/截断 | 原 helper 没有统一的 mismatch EOS guard | 本 benchmark 统一阻止 EOS mismatch，以及可见 token 预算外的 relaxed mismatch；exact token 优先级保留。属共享停止安全适配。 |

“原样保留”仅指明确映射的机制/公式，不表示整体模型、视频输入、Draft pruning、K10、512-token、H200 等原始实验设置已经复现。

## 参数与开发集选项

`PAPER_DEFAULTS` 的完整可见参数：

```json
{
  "loose_ratio": 0.7,
  "similarity_top_n": 10,
  "draft_window": 10,
  "pst_enabled": true,
  "pst_include_root": true,
  "visual_reference": "image_tokens",
  "stable_rank_ties": true,
  "disable_relaxation": false
}
```

主臂建议名称为 `LVSpec-ViSpec-paper-ratio`（固定 λ=.7）与 `LVSpec-ViSpec-adapted-ratio`（仅独立开发集在 `{0,.3,.5,.7}` 选 λ）。固定 N10、max span10、PST 与其它策略；不根据主测试质量或标签选择规则。若所有开发配置均不满足预先声明质量条件，则独立启用严格 fallback、明确改名。λ0+PST 不可充当该 fallback。开发/测试各样本三次生成的评价与计时规则由全局协议冻结，此模块不读取结果决定配置。

有限短树会导致多个 λ 映射到相同低排名位置数量。它们是否最终等价须依据每个任务完整生成判断；不能因某次主测试表现不理想增加搜索范围。

## runtime、KV 与数据范围

- Controller 只包装新 fresh ViSpec runtime 的四个入口；所有 Target/Draft 调用、`position_ids`/mRoPE、causal tree mask、KV commit 和采样 next-token 原函数原参数执行。
- final-norm hook 被动捕获当前原生 Target forward；没有为 LVSpec 新增 Target forward。prefill 只保留真实视觉行，释放整段捕获；tree 只保留当前真实节点状态，update 后释放。
- audit 模式独立核对 committed prefix、真实 tree mask 祖先集合、position depth 与 candidate/native node token 对齐。不通过 fabricated 普通 causal mask 重算验证。
- relative rank/PST 只读取已获得的本轮草稿树与当前原生 Target 结果，不读取最终 Caption、任务参考答案、Judge 或 Harmful 标签。分支后续 Draft 位置是已存在的 speculative proposal，不能误写成“仅当前单 token 内禀特征”。
- 同 parent 节点绝对相关性可复用，但 per-branch 相对排名不跨不相关 prefix 复用。本实现先算每个真实 node 的相关性，再对各行分别排序。
- 接受路径保存 image/event 外围 runner 标识及每轮 row、node/token、接受长度、mismatch 与已提交累计；反事实标签与来源不参与接受规则。
- terminal update 可复用已在 CPU 的选中 path token hint；原生停止与 KV 提交仍由共享 wrapper 执行。
- restore 恢复 runtime 原入口并移除 norm hook；禁用循环 GC 的 CPU 测试验证 controller 可立即析构。

## 验证证据

执行：

```bash
/data/wbb/miniconda3/envs/spec/bin/python -m unittest discover \
  -s tests -p 'test_lvspec*.py' -v
```

本代理实测 **12 项 CPU 测试全部通过**，没有 GPU 调用：

1. 与 fresh clone 中真实 `visual_similarity_hidden_state` AST 函数计算核对，N 大于视觉行数时正确取 min；FP32 误差容限 `atol=1e-6,rtol=1e-5`。
2. 100 组、每组3条 K10 chain 的接受长度与官方 `loose_match_hidden_topk_or_search(k=7)` 直接核对。排除本次额外 EOS/截断 guard 的触发以核对原规则。
3. 合法树行的局部排序、共享 parent 的不同 branch rank、不受 unrelated branch 改值影响；拒绝混合祖先、重复 node 与中间 padding。
4. λ0+PST、root inclusion、显式 disable_relaxation 与无 PST 的区别。
5. PST 只查同 branch、最大 span10；不读取 sibling 或 span 外 token。
6. EOS、预算、padding、原生 exact 优先级；LVSpec 可接受超过2/4个 mismatch 的实质预算差异。
7. actual branch length 的 floor 比例映射与可复现 tie 行为。
8. 真实 runtime 风格 final-norm hook，确认视觉行筛选、parent位置、输出 logit vector、无额外 Target调用。
9. explicit Strict fallback 与原 greedy posterior 的选择和原生 commit 一致。
10. tree mask 与 candidate/node audit 失败可见，T≠0 processor 不被静默当成 greedy。
11. hook restore 与无引用环立即析构。

尚需整个 benchmark 的真实模型 parity：native Strict、共享 plumbing Strict、LV strict fallback 在相同输入下输出核对；真实图像 prefill/树状态的 shape、dtype、mask/position 与 model.norm 捕获一致；主测重复确定性、峰值内存及运行成本。CPU 通过不能代替这些实证。

## 可以与不能支持的结论

可以比较此**已公开适配设置**下，LVSpec 的相对视觉+PST 策略与其它方法的完整生成质量、接受长度及实际成本。不能以绝对 V1 cosine 的信号 AUROC 代表 LVSpec，也不能把本次 image-only/Qwen7B/ViSpec 树/128-token/静态图像的结果称为原视频论文完全复现。

LVSpec 没有共享 BEPC/ME 的2/4 cap，因此主表衡量实际方法策略包的质量—速度结果；不是只改变一个门的严格预算匹配消融。多分支的 longest selection 会改变松弛程度，PST 还可能放行视觉排名较高的 mismatch；“高相关 token 必须严格”并非包含 PST 后的绝对不变量。短输出任务、否定/数量/答案字母等低视觉相关但关键 token 也不能由该启发式保证安全。[正式论文 Appendix E/I](https://aclanthology.org/2026.acl-long.1087.pdf)

独立开发集约束和自动质量指标不构成人工事实安全 Ground Truth；此模块不训练任何验证器、不融合 BEPC 组件、不依赖测试标签。

## 2026-11-01：实际GPU与最终交付证据

上文“尚需GPU parity”描述的是实现阶段状态。现已完成120条真实GPU parity、3240条开发测量和12600条主测试测量；三次生成输出差异为0，原生ViSpec与strict_shared在全部主测问题上的返回token、接受路径、EOS、截断和Target调用均一致。独立全量审计共18项检查，所有必需项通过，required_failures=0；实际量化结果见[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)，证据见[PARITY_GATE](../audit/PARITY_GATE.json)、[统计封印](../audit/STATISTICAL_ANALYSIS.json)、[最终审计](../audit/FINAL_DELIVERY_VALIDATION.md)。

开发集保留MME/GQA/ScienceQA的lambda=.7；TextVQA与COCO全部四档未通过预设代理质量门，测试适配臂明确disable_relaxation=True。未使用测试结果改阈值。所有主测结果（包括GQA/文字识别/Caption质量下降与较慢结果）均完整保留。历史全未见图像泛化主张仍为No-Go；本轮dev/test物理图像独立核验通过。
