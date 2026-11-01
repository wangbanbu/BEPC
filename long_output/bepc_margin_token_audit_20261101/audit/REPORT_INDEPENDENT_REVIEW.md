# Final report independent review

Status: **PASS with documented auxiliary matching limits**. The read-only review checks the report against frozen production outputs, all276 actual-visited v2 diagnostics, paired official scores, COCO review distributions and derived data deliveries.

Verified:300 paired questions;276 changed outputs;83,236 raw tokens;3,337 changed spans;QA4/19 surfaced-answer changes and14 score changes;126 direct Target-retention cases;190 protection and114 rescue actual gate flips;84 protection cases at native margin1;3,426 Target/12,600 Draft/276 visual prefix-replay calls. All five highlighted gate numerical examples agree with stored diagnostics.

The two new parquet files exactly match their v2 JSON sources. All324 mapped candidate-string spans match saved original character/token slices;41 entries reflect literal matcher non-hits. All13 delivery-index paths exist, diagnostic CLI --mode changed is valid, and all14 source-freeze hashes match.

No unsupported claim of a single-token task-quality causal effect, human Ground Truth, or established BEPC superiority was found. Explanation-body forks, output-format failures, subword ambiguity, later generated hallucinations, metric-enriched selection and missing independent hidden/logit tensors are explicitly acknowledged.

Auxiliary matching caveat: Updated matching distinguishes single-uppercase letters, decimal/thousands prefixes and literal punctuation. It remains a literal candidate index, not a typed full-semantic-unit parser; option/sample-letter roles and semantic relevance require contextual review, as the report now states.

Reproduction commands assume execution from this audit root. This review did not run generation, invoke a judge, or overwrite audited inputs.

Report SHA256: `a3cbc2c06654e611d952355263f0946f6565b6cde2446c3d2388c0ca27ef5aca`. Generator SHA256: `7d0548e67581996dee8bbdb3e45bdce5aebaf4fd8f5dc906ca8f0b2c6d7c6fd4`.
