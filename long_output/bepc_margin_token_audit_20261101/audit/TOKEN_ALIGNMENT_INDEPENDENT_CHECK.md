# Independent token/path alignment check

Status: **PASS with interpretation limits**. Independently reconstructed all600 request ledgers from recorded selected paths and checked all300 pairs,83,236 raw ledger tokens and3,337 changed spans against saved parquet outputs. The independent checker did not call generation, recompute official metrics, or overwrite audited inputs.

The300-pair counts reproduce exactly: ScienceQA87 changed outputs, TextVQA91, COCO98. All600 path flattenings exactly reproduce raw token IDs; returned IDs are the visible prefix; decoded text matches production text. First token pieces, origin roles, round indices, common token prefixes, span IDs/bounds, token IDs and span text all match. QA surfaced/normalized answer counts and official score deltas agree with frozen score records.

All14 source-freeze hashes, 29 parent delivery artifacts, 12 scoring seals and 262 prior preservation entries match. Audited extraction source and outputs remained byte-identical throughout this review.

Post-fork exact-token alignment occurs in 276 changed pairs (3294 later equal spans). These repeated equal words do not restore a common causal prefix. The LCP boundary is the only uninterrupted shared emitted-token prefix.

All23 surfaced final-answer changes first diverge in their explanation body. This does not mean later final-answer differences were directly protected, because trajectories had already separated. Format-specific score changes34741/37671/38948 were independently verified to terminate at EOS without truncation.

Interpretation limits:

- first_selected_path_divergence_round is a path/block statistic; it is not necessarily first visible token divergence or the causal gate difference.
- target_root versus draft_node denotes a token origin/block role. A draft_node need not be an accepted mismatch; exact matches are also Draft nodes.
- The first_change_in_final_answer_line flag uses a decoded substring boundary and rfind, not the frozen answer parser. All23 surfaced-answer changes have clearly earlier body forks here, so the current aggregate is correct; use parser/offset-aware matching before generalizing the flag to arbitrary outputs.
- SequenceMatcher is deterministic alignment, not minimal edit distance or causal intervention alignment. alignment_change_size is a descriptive alignment size, not number of direct BEPC decisions.
- The audit verifies recorded paths reconstruct original tokens; it does not reconstruct tree causal attention, unseen rejected nodes, or candidate gate scores.
- Different final answers and official score deltas do not identify a direct protection gate. Same-prefix candidate diagnostics are required.
- All hashes matched before/after independent check; historical raw outputs, score files, and sealed sources were not modified.
