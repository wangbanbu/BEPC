# 长输出独立交付审计

Status: **pass**；phase: `preflight`。CPU-only，无 GPU、Judge 或训练调用。

| Check | Result |
|---|---|
| fresh_source_and_readonly_adapter_provenance | PASS |
| development_configuration_and_checkpoint_identity | PASS |
| data_images_prompts_labels_history_and_groups | PASS |
| all_failures_retained_and_resolved | PASS |
| repeat_determinism_and_claimed_equivalence | PASS |
| strict_project_unseen_image_claim | No-Go: AssertionError('Strict all-project-unseen condition is No-Go; declared historical overlaps retained without resampling') |

逐请求 MAT/停止/模型调用、完整网格、输入与gold封印、确定性、评分和统计证据见相邻JSON。历史未见图像主张单列，不等同执行失败。

## 局限

- MME history reuse limits unseen-image generalization.
- Seven additional non-MME aliases are historically seen; the strict all-project-unseen claim remains No-Go.
- 20/100-image subsets and automatic metrics do not establish population noninferiority or human factual safety.
- Checkpoint bytes use a preserved prior SHA manifest plus current stat tuple; this audit does not rehash all checkpoint shards.
- Production ancestry correctness relies on sealed source, CPU contracts and explicit small GPU parity; full hidden-state replay is not rerun.
- This validator audits the delivered bootstrap inputs/draw seals; a second implementation of every confidence interval is not claimed.
- All images reuse the declared short-output cohort; long/short changes prompt AND budget, not a pure prompt-only causal experiment.
- Terminal QA accuracy does not evaluate explanation truthfulness; full long-description CIDEr/SPICE/CHAIR do not constitute human factual Ground Truth.
- AR/native Strict token divergence is separately reported, not silently assumed lossless.
