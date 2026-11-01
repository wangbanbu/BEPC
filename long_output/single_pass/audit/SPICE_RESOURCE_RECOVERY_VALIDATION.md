# SPICE资源恢复独立审计

Status: **pass**。仅CPU文件审计；不执行Java、GPU推理或重新生成。

| Check | Result |
|---|---|
| successful_CPU_recovery_manifest_and_only_heap_change | PASS |
| unchanged_official_JARs_wrapper_models_and_classpath | PASS |
| preserved_failure_and_identical100_caption_input | PASS |
| original4200_GPU_records_runtime_config_successful_metrics_unchanged | PASS |
| complete_new_output_no_ignored_parse_and_sealed_log | PASS |
| same_official_scorer_output_to_final_scores_and_seals | PASS |
| active_failure_scan_excludes_preserved_audit_archive | PASS |

实际producer采用64G JVM heap；冻结wrapper在精确input的cache hit中仍展示nominal16G计划命令，两者由资源恢复manifest明确区分。原16G失败日志、输入和部分缓存保留在audit归档；失败评分未混入最终结果。

## 局限

- The audit independently checks preserved files, command differences, logs, IDs and score joins; it does not independently rerun Stanford parsing or certify every extracted scene graph.
- Fresh-cache emptiness at start is attested by the sealed recovery implementation and recorded runtime state, not by a separate real-time observer.
- The frozen official wrapper reports its nominal16G command on a cache hit; the resource manifest and preserved log/output identify the actual64G producer.
- Long descriptions and automatic caption metrics retain their previously reported validity limitations.
