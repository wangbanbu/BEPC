# Independent implementation review

Status: **PASS — no blocking implementation finding.** This review covers the code and frozen paired design; completed-run validation and scientific results remain separate.

The comparison contains only `bepc_no_response_cap` and `margin_no_response_cap` on ScienceQA, COCO and TextVQA. Both use tau=1, round mismatch budget 2, no response-wide cap, identical native tree/KV/mRoPE/mask/stopping, compact logits and terminal CPU selection hints. The frozen configurations are authoritative. Historical Strict results provide context and are not a concurrent third arm.

Plain Margin uses native Target top1 and Draft candidate logits, casts each to FP32 before subtraction, rejects nonfinite values and accepts g<=1. The factory routes T2 to NoResponseCapMargin. Its parent installs no visual/text/Draft observers; all three observer fields are asserted None. Both arms retain the same SpeedController scanner/update and first-longest selected-path accounting. Init-only response4 is replaced by an infinite live budget before any forward or decision; serialized budget is null.

The independent audit already ran the three new tests: **3 passed, 0 failed** (reported runtime 0.004 s). Coverage includes compact/expanded FP32 margin identity, threshold and nonfinite handling, no representation hooks, first-longest ties, per-round limit and selected-path accounting beyond four total mismatches. Tests were not rerun while persisting this review.

Generation requires exact historical BEPC input, raw/visible tokens and selected-path parity. The validation scripts check paired inputs, record counts, round2/no cap binding, native Target/vision call counts and cleanup. The pipeline starts official CPU scoring after GPU generation exits; main scoring is called by analyze_comparison.main. These are implementation checks; their completed-run evidence is still pending.

Non-blocking notes:

- BEPC's inherited `all_relaxed_methods_same_budget=False` describes the previous cap4 comparison. It is stale, non-operative metadata here: both live response budgets are infinity, and both authoritative frozen arm configurations specify round2/no response cap. No decoding-code change is required.
- On worker failure the pipeline raises without terminating an unfinished peer. Root monitors and handles failure cleanup manually.
- Same threshold and budgets constitute a fixed-rule ablation. They do not establish equal acceptance rate or a quality-matched speed optimum. The protocol retains the single-pass timing, reused-image and automatic-quality limitations.

Exact reviewed source hashes, frozen arm configurations and detailed checks are in `IMPLEMENTATION_REVIEW.json`. Only these two audit artifacts were written by the reviewer; inference sources remain unchanged.
