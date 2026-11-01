# 单次长输出实验独立交付审计

Status: **pass**；phase `all`。CPU-only。

| Check | Result |
|---|---|
| preserved_original_run_and_label_blind_inventory | PASS |
| fresh_official_sources_and_readonly_originals | PASS |
| development_configuration | PASS |
| main_configuration | PASS |
| exact_same_inputs_and_official_labels | PASS |
| reuse_original120_GPU_parity_no_new_requests | PASS |
| development_actual_imports_GPU_exclusivity | PASS |
| development_one_pass_grid_origins_MAT_calls_budgets | PASS |
| main_actual_imports_GPU_exclusivity | PASS |
| main_one_pass_grid_origins_MAT_calls_budgets | PASS |
| development_official_quality_and_seals | PASS |
| main_official_quality_and_seals | PASS |
| development_only_threshold_selection | PASS |
| grouped5000_bootstrap_points_Holm_output_seals | PASS |
| failures_retained | PASS |
| strict_shared_native_equivalence | PASS |
| project_history_unseen_claim | No-Go: AssertionError('Strict all-project-unseen condition is No-Go; declared historical overlaps retained without resampling') |

本轮每题/方法仅一次测量；不能据空的重复差异列表声称已验证重复确定性。完整源库存、来源映射、MAT、调用预算、官方评分和统计封印见JSON。

## 局限

- Only one measured request per method/question; run-to-run timing variability and repeat determinism are not assessed.
- 5000 confidence intervals concern paired image/group sampling, not independent repeated timing runs.
- Development reuse spans original and resumed sessions; source measurements and preprocessing differences remain explicit.
- Long QA answer accuracy does not verify explanation truthfulness; automatic caption metrics are not human factual Ground Truth.
- Same existing images as the short-output cohort, including all declared historical overlaps; no unseen-image generalization claim.
- Token/call/MAT point estimates and grouped bootstrap draws/Holm correction are independently checked; every interval is not independently reimplemented.
