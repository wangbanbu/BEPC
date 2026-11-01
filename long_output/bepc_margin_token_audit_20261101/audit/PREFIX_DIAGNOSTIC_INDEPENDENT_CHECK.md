# Independent first-fork prefix diagnostic check

Status: **PASS with interpretation limits** for all276 changed-output cases. No generation was run by this reviewer. Original result, metric and source files remained unchanged.

The checker independently reconstructed552 production/shadow branch scans and their visited gate sets, winner row, accepted lengths and mismatch counts. Every BEPC path matches its historical first-fork path; every same-tree Margin shadow matches the historical Margin path. All candidate ancestor prefixes, token positions, token pieces, native-margin row duplicate alignments, selected edges and legal-prefix hashes are consistent.

FP32 correction and score recomputation from stored components agrees exactly (maximum delta error 0.0). This verifies the arithmetic from stored values, not an independent hidden-state projection or full-logit recomputation.

Prefix replay cost is **3426 Target +12600 Draft +276 visual forwards**. The field `extra_diagnostic_target_forwards=0` refers only to supplementary readouts after these forwards; it must not be interpreted as no repeated Target work.

Across first-fork trees, 192 protection and 119 rescue flips were stored. Of these, 7 flips were supplemental and not visited by production BEPC. 0 original direct-first-visible classifications involve an unvisited BEPC candidate.

Candidate-flip counts are conditioned on276 output-changing trajectories, not harmful-event rates or natural distribution prevalence.

Interpretation limits:

- Production BEPC-visited flags were captured before supplementary full scores and exactly reproduce independent per-branch scans. Margin shadow visits and selected path also exactly reproduce stored Margin historical paths at the first common-tree fork.
- All stored candidate legal prefixes consist exactly of original prompt plus previously committed common output plus that candidate parent true ancestor path, with candidate positions, hashes, duplicate row logits/margins and subword pieces consistent.
- The new audit DID perform3426 Target,12600 Draft,276 vision forwards to reconstruct prefixes. Only additional score readouts on already computed first-fork hidden states cost zero further Target forwards. Do not call this entire audit zero-forward.
- Gate delta and score were independently recomputed from stored m_U/m_P/native g with FP32 arithmetic. The raw hidden states and raw logits are not saved here, so this reviewer does not independently rederive m_U/m_P from tensor projections or native g from full logits; those depend on frozen runtime implementation and historical path/input parity.
- selected edge and first visible token are distinct: a deeper changed gate can switch the selected ancestor branch and alter an earlier visible token.
- Protection/rescue denotes acceptance decision direction, not whether it improves factual/task quality. All final outputs used to examine consequences come from the frozen previous experiment.
- Subword token pieces are not complete semantic units. Output-level spellings/digits must not be attributed to one rejected piece without a reconstructed semantic span.
- All276 are first divergent-tree audits; later gates and later quality changes are outside attribution scope.
- Unvisited supplemental flips are hypothetical under the alternative scan and must remain distinguished from actual production BEPC gate decisions.

## Strict actual-visited v2 cross-check

Independently verified all276 v2 cases, all304 actual BEPC-visited flips (190 protection,114 rescue), seven excluded shadow-only hypothetical flips, and every task summary count. All raw component fields agree with native diagnostic files. No direct-first-visible classifications change from v1 because none of its direct classifications involved an unvisited BEPC gate.
