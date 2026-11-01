# Independent uncapped BEPC implementation review

Status: **PASS_PRELAUNCH_CORE**. This audit ran no GPU inference and changed no existing method source or prior artifact.

- **single_factor — PASS.** Only active self.response_budget changes from 4 to positive infinity. Symmetric gamma .5, tau1, U8/P4/W32/pool64, kappa1, eta1e-4 and residual attenuation off remain frozen.
- **bootstrap_init_only — PASS.** online Controller/RefinedController/SpeedController/ExperimentalBEPC constructors only initialize state/install hooks; no original initialize_tree/tree_decoding/update calls. math.inf replaces bootstrap4 before any input forward or candidate decision.
- **old_score_path — PASS.** Equal rescue/protection scales use SpeedController.allowed verbatim, unchanged native margins, shortcut bounds, full score and legal parent contexts.
- **branch_and_round_rules — PASS.** Frozen budgeted_scan retained. finite committed+branch mismatch count cannot reach infinity; round budget2 and all padding/EOS/visible-boundary logic unchanged. First-longest tie rule and selected-path-only commit unchanged.
- **observer_lifecycle — PASS.** ExperimentalBEPC.evaluate cap-exhausted fast path never enters; update retirement condition finite count >= infinity never true. Observer stays alive until normal restore.
- **statistics_and_json — PASS.** final_stats overwrites inherited infinity with JSON null before caller serialization; explicit no-cap/retirement false fields. Committed counters remain integers. json.dumps allow_nan=False exercised at 12 mismatches.
- **inherited_forward_and_stopping — PASS.** Inherited run_benchmark generation, original native Target/Draft, tree/KV/update, max_new_tokens/EOS and terminal-update implementation are byte-identical copied files; no new overrides.
- **concurrent_controls — PASS.** run_uncapped factory selects uncapped subclass only by explicit config flag; capped/strict controls checked byte-for-byte against prior raw and returned tokens, input, acceptance lengths and paths.
- **cpu_tests — PASS.** 6/6 tests passed in spec Python with CUDA_VISIBLE_DEVICES empty. Includes 10,000 independent randomized scan checks, 12 committed mismatches, no retirement, round2, off-path accounting, capped4 control and parameter guards.
- **previous_preservation — PASS.** All 213 prior file SHA256 entries match; ten key inherited files are real byte-identical copies, not symlinks.

## Remaining evidence required

- No GPU numerical/decoding claim established by this review; concurrent capped/strict historical parity must pass at runtime.
- Frozen config/protocol/entry source seal must be finalized before generation; this core review does not approve an unseen configuration.
- Removing the response cap can increase cumulative relaxed decisions and active BEPC feature overhead. Speed and quality are empirical and are not guaranteed.
- No human factual-ground-truth conclusion; quality conclusions require official task evaluation and uncertainty reporting.
