# Independent uncapped BEPC metric/protocol review

Status: **PASS_PROTOCOL_AND_METRIC_IMPLEMENTATION**. No result superiority is inferred. Reviewer ran no GPU forward or official scoring and edited no runtime source.

- **frozen_single_factor: PASS.** Each of three task configs differs between cap4 and uncapped only by response_mismatch_budget 4->null plus explicit no_response_cap=True. Cap4/Strict are exactly prior frozen method dictionaries; symmetric .5 remains, not .625.
- **runtime_null_inf: PASS.** Init-only bootstrap4 compatibility verified in core review; positive infinity disables only response-wide comparisons before any forward. Final JSON budget null, counters intact; no finite hidden response cap or observer retirement.
- **frozen_sources: PASS.** All23 generation runtime SHA and6 official-evaluator/analyzer/parser source SHA match. design_config bytes equal frozen_config. Metric code sealed before quality scoring.
- **same_data_and_groups: PASS.** Each task has20dev and100test groups, all physical image SHA unique within split; new run uses same prior frozen inputs/prompts. Previously exposed data explicitly declared paired retest, not unseen.
- **staged_gate: PASS.** Pipeline waits all6 dev completion markers, no GPU pids, then independent validate --phase development before main launch. Dev integrity does not tune quality thresholds. MAIN_READY states dev integrity requirement.
- **runtime_integrity: PASS.** Validator checks full unique grids, result/config/source seals, same inputs, repeat0/T0, Targetround+1 and vision1, no extra Target, lifecycle0, round2 selected-only accounting, >4 allowed and no uncapped retirement, historical capped/Strict tokens+paths equality, first policy path divergence cannot precede committed>=3.
- **gpu_scoring_separation: PASS.** Orchestrator checks both main subprocess exits0, validates main, asserts no GPU pids and only then launches CPU development scoring and main scoring/analysis. This excludes official Java/CPU scoring overlap during this run's GPU measurements.
- **official_scores_and_reuse: PASS.** Read-only prior final-answer parser, official ScienceQA/VQA/COCO scorers. Reuse requires exact whole ordered labeled corpus, raw output text and parser/evaluator signature, source score SHA, IDs+raw text double check. No per-caption cherry-pick CIDEr reuse or reused old timing.
- **spice_resources: PASS.** New SPICE official same jar/models/options with64G heap, fresh experiment cache and serialized lock. Failure/ignored sentence logs block seals; full unique IDs required. Classpath hashes recorded. LMDB persistence warning distinct from omitted parse.
- **statistical_math: PASS.** Same5000 paired physical-image bootstrap draws shared across arms; unit verified group==one uniqueimage for these tasks. Ratio-of-sums throughput/MAT, all3 prespecified pairs/task/metric Holm, no global multiplicity claim, p approximate centered bootstrap and CI omit repeat noise.
- **quality_claim_limits: PASS.** NI tolerance predeclared inherited QA2pp, SPICE95percent, CHAIRs2pp; paired one-sided bounds, each reference separately. Human caption statuspending, zero-discordance bootstrap caveat, answers do not validate explanations. No automatic-metric humanGT or factual safety claim.
- **all_outputs_retained: PASS.** All900test/180dev planned irrespective quality; report saves allchanged outputs/score transitions, EOS/caps/parser failures and budget/cost counters. No new classifier/fusion/label rule or parameter selection.
- **cpu_selftests: PASS.** evaluate_uncapped --self-test and analyze_uncapped --self-test independently passed with CUDA hidden; parser, immutable identities, paired stats, tiny CIDEr formatting, NI and uncappedround2 diagnostics. No new official scoring executed by reviewer.

## Final evidence still required

- Final generation completeness, actual historical parity, absence of resource overlap, score provenance/seals and report values require postrun validation.
- Main entry can technically be run directly using prepared MAIN_READY; this review relies on the audited orchestrator enforcing dev validation. No direct main launch performed.
- Pipeline and validator helper source SHA captured here in addition to runtime/metric seals; final delivery must recheck these hashes.
- Single-pass clocks and previously exposed100test groups/task do not establish general or repeatable speed-quality guarantee.
- Automatic caption metrics remain auxiliary evidence; no independent double-human annotations were added.
