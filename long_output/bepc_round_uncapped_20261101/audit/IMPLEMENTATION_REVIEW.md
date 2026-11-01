# Independent BEPC round-cap removal implementation review

Status: PASS — preflight implementation only; GPU results and generation quality are pending.

The new arm removes the per-round mismatch count limit from the already response-uncapped BEPC. Its frozen configuration differs from the concurrent control only in `round_mismatch_budget: null` and `no_round_cap: true`. Symmetric gamma=.5, tau=1, U8/P4, text window32, pool64, kappa1, eta1e-4, residual attenuation=false and both-direction correction are unchanged.

`NoRoundCapBEPC` uses the old constructor's round2 only during initialization, then sets the live instance budget to positive infinity before any request forward or acceptance decision. `SpeedController.evaluate` passes that instance value to `budgeted_scan`; no second operative per-round cap was found. The response-budget fallback and feature retirement remain unreachable. JSON-facing statistics override the parent's hardcoded round2 and serialize both uncapped budgets as null.

All 26 inherited runtime files match the previous frozen source manifest byte for byte; the frozen set contains 29 runtime files including the three new entry/controller/preparation files. All 232 previously preserved files rehash unchanged. The subclass changes neither Target/Draft forward nor the original score, native logits, first-longest branch rule, EOS/visible-boundary handling, KV commits, attention mask, mRoPE or output stopping.

The real Draft implementation first produces one layer and then performs `depth=3` additional expansions. Thus the longest branch has at most four Draft descendants plus its root. The 30 Target rows comprise 29 selected Draft nodes plus root. Uncapping changes eligible acceptances within this existing tree; it does not add tree depth. Root tokens are excluded from mismatch and MAT counts.

Observer histories remain legal for third/fourth mismatches: the current parent receives only committed states plus its actual ancestors, and arbitrary selected path lengths are committed without assuming two mismatches. Siblings and future descendants are excluded.

Three CPU unit tests passed independently and in the retained log. They cover four accepted mismatches only in the new arm, the real inherited finite-margin gate, repeated selected-path commits, no off-path budget spending, EOS and visible boundary, first-longest selection, lifecycle restoration, strict JSON serialization, and 10,000 randomized comparisons with an independent scanner definition.

The first-path-divergence check is logically appropriate. With equal common history, every branch accepting at most two mismatches has the same acceptable length in both arms. Consequently a newly selected path that first differs must contain at least three accepted mismatches, including when first-longest selection switches leaves. The runtime validator checks this, plus exact historical control paths, same-input pairing, selected maximum four, 30 native rows per round, zero extra Target forwards and no live request objects. Before generation, the validator was also strengthened to reject unequal path-list tail lengths or changed raw tokens when all zipped common paths match. The reviewed source hash includes this correction.

Generation and scoring are sequenced: development integrity validation precedes main inference, then both GPU workers must exit before CPU official metrics. The 600 main and 120 development request counts, actual >2-mismatch frequency, final parity and output-quality comparisons remain empirical checks for the completed run.

No runtime files were modified by this review. Detailed evidence hashes and checks are in `IMPLEMENTATION_REVIEW.json`.
