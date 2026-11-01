# MAT、长输出提示词与 AR/Strict 对齐审计

审计日期：2026-11-01（Asia/Shanghai）。只读核对已冻结短输出记录、fresh upstream `49a33ff0ca31e281b9dc8803fd6f3041c6c0e135` 及旧 COCO 长输出记录；没有重跑或修改已有答案。

## MAT 的可复核定义

官方 [`evaluate_posterior`](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/model/utils.py#L414) 在 T=0 下比较 `candidates[:, 1:]` 与预测它们的 `logits[:, :-1].argmax`。返回的 `accept_length=a_r` 是第 r 轮连续接受的 Draft 候选数，**不包含每轮已经由 Target 确定的 root token**。官方 [`update_inference_inputs`](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/model/utils.py#L495) 提交 `a_r+1` 个 token；[`spec_model_ours.py`](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/model/spec_model_ours.py#L505) 原样保存 a_r。

官方 [`speed.py`](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/speed.py#L66) 将所有请求的 `acceptance_length` 列表合并，计算：

\[
\mathrm{MAT}_{native,draft}=\frac{\sum_r a_r}{R_{native}}.
\]

这是轮数加权的 corpus 比率，不是先求每请求 MAT 再简单平均。它可能包含最后 block 的 EOS 后尾部或超预算尾部；这些 token 不属于可见输出。本实验保留此原生数值，同时以有效 Draft MAT 作为主要接受长度：

\[
C=\min(B,\text{首个 EOS 位置}+1),\quad b_r=\sum_{k<r}(a_k+1),
\]

\[
a_r^{eff}=\max\{0,\min(a_r,C-b_r-1)\},\qquad
\mathrm{MAT}_{effective,draft}=\frac{\sum_{r:b_r<C}a_r^{eff}}{|\{r:b_r<C\}|}.
\]

没有预算内 EOS 时 C 为实际返回的预算内 completion 长度。EOS 若落在 Draft 接受区间，仍是一次真实接受；可见内容 token/s 则排除 EOS。另报含 root 的推进长度 `1+MAT_effective,draft`，便于区分部分文献采用的 token/iteration 口径。AR 没有 Draft 验证，MAT 为 NA，不能把 runner 的零占位解释成接受能力。

停止规则维持原生 `new_token > max_new_tokens`、block 结束后检查 EOS。因此可能实际计算预算以外 token；计时包括这些真实成本，评分统一先截 B 再取首 EOS 前内容，不以清洗后长度重算原生调用数。保留 raw/returned token IDs、EOS、cap_reached、有效轮数与原生轮数，能独立重建全部分子分母。

## 速度口径

主吞吐为预算内首 EOS 前**内容 token**总数除以 CUDA 同步模型请求总时间；模型请求包括 controller 构造、视觉编码、prefill、Target/Draft 和在线信号。另报含 EOS 的 completion token/s、decode-only token/s、每请求时延、内容长度、EOS 率与截断率。PIL/processor/H2D 独立计时并在配对方法间复用，不称为完整客户端端到端时延。

重复测量先在每问题内平均分子/分母，再按 image/group 聚合比率；同一配对 bootstrap 抽样索引用于全部方法。长输出变长或触顶能提高 token/s 却增加整题时延，两个指标必须并列。

## 官方长输出先例与本轮适配

官方 [MME](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/mme_prompt.py)、[GQA](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/gqa_prompt.py) 在问题后追加 `Please answer with an explanation.`；[ScienceQA](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/scienceqa_prompt.py#L184) 要求以 `The answer is` 开头并解释；[COCO](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/coco_caption_prompt.py#L26) 要求详细描述图像。官方 [TextVQA](https://github.com/KangJialiang/ViSpec/blob/49a33ff0ca31e281b9dc8803fd6f3041c6c0e135/vispec/evaluation/textvqa_prompt.py) 追加详细 OCR 过程指令，会改变原来的问答任务，不直接用于本次任务质量比较。

五个官方 `gen_spec_answer_*` 入口的 `--max-new-token` 默认 1024；底层 `specgenerate` 默认 512、max_length 2048。旧 `signal_decoding_long_output/configs/frozen_config.json` 已实际使用 detailed COCO、512-token 预算；它是历史先例，不是当前五任务结果。

本轮决定：全部任务统一 512-token 预算、max_length 2048，原模型、树、图像处理与样本不变。QA 要求约 100–160 词解释**后**给出唯一 `Final answer:` 字段，减少答案先由共同 root 确定而后续宽松规则只改解释的评估盲区；COCO 要求约 100–160 词的详细可见图像描述。确切提示和固定 parser 由 `reports/QUALITY_PROTOCOL.md`、配置与输入文件 SHA 确定。此处是**研究适配**，不归因于官方论文或宣称原样复现其 prompt。

最终答案截断、缺失、重复冲突或提取失败均保留，QA 记零并另报格式失败率。QA Accuracy 只能评估提取的最终答案，不能证明解释事实正确；COCO CIDEr/SPICE/CHAIR 仍不能代替人工事实/覆盖 Ground Truth。100–160 词是请求长度，不强制生成到该长度、不补采或删除短答。

## 已有 17 题 AR/Strict 差异：证据与未解部分

短输出独立审计已证明：600 题中 AR/原生 ViSpec 的文本与返回 token 有 17 题差异（TextVQA 1，COCO 16）；原生与共享 Strict 600 题完全一致；三重复输出完全一致。这是保留结果的事实，不是差异机制的证明。详细原始记录在父目录 `audit/AR_STRICT_OUTPUT_COMPARISON.json`。

源码能够确认：

- `harness/run_benchmark.py::serial_ar` 使用同一个 custom Target 和 KV 对象，清零长度、reset_tree_mode，prefill 一次，随后串行输入实际生成 token。
- custom Qwen forward `modeling_qwen2_5_vl_kv.py:2138–2178` 在非空 KV 时自动设 cache_position，并加保存的 rope_deltas；现有 `test_runner_contract.py` 执行实际源码 AST 验证该逻辑。因此，“AR 未显式传 position_ids”本身不是已证明的错误。
- 原生 `utils.py::tree_decoding` 显式使用 `prefix_length+tree_depth+rope_delta`；update 根据 retrieve_indices 将已选路径的 KV 拷回连续区间。
- 实测 Target dtype 为 BF16、attention 为 `Qwen2_5_VLSdpaAttention`。单 token 与多 token 树的 GEMM/SDPA 张量形状不同，单 token还可能走无显式 causal mask 路径；存在数值差异的合理可能性，但尚未测得首个分叉的 logits/误差来确认。
- 树 causal mask 与 KV gather 的正确性不能仅靠 native/shared 输出一致证明；两者共享同一原生实现。

建议仅在少量长输出开发样本上诊断：保存同一输入 prefill root、首个分叉前共同 prefix、该处 native tree 与独立串行 forward 的 top-2 logit margin、候选 logits、max-abs logit 差、position_ids、cache.current_length、retrieve ancestor 集及有效 mask。先看 root 是否相同，再判断是微小数值翻转还是明显位置/mask/缓存错误。只在开发集做定点核对，不根据测试质量改变阈值，不将诊断调用混入正式计时。

在定位前，继续报告 AR/Strict 的实际不同质量和速度，不能宣称完全 lossless 或逐 token 等价。新长输出 native/shared 与工程 reference parity 仍须通过，三重复差异须完整保留。若出现明确缓存或 mask 错误，依其影响范围停下正式比较并修复，不能用“BF16”解释替代证据。
