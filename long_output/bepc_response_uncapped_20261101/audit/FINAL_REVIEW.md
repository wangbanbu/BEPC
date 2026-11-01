# Independent final review: uncapped response-budget BEPC

**PASS** for complete and accurately scoped pilot delivery. Reviewer ran no GPU forward or official scorer and changed no frozen code, results or report.

- **single_factor_implementation: PASS.** Original symmetric BEPC score and allround2/tree/KV/position/attention/stopping rules unchanged. Only response cap4->none. Init-only compatibility4 never reaches a forward/decision. No hidden finite cap or observer retirement in uncapped.
- **full_generation_grid: PASS.** All180 development and900 main records present uniquely, repeat0/T0; independent main accounting confirms vision1/Targetround+1/no extraTarget, visible Draft+round accounting, no lifecycle leaks.
- **control_reproducibility: PASS.** Independent exact input/raw+visible tokens/acceptance length/path comparison confirms720 concurrent capped/Strict requests match previous source:600test+120dev.
- **intervention_timing: PASS.** Independent checking confirms344 actual first divergent selected paths across dev+test have shared committed count>=3; no premature divergence before budget4 could bind under round2.
- **uncapped_really_uncapped: PASS.** Main uncapped counts: SQAmean10.06,max23,over4=89/100; COCOmean27.34,max60,over4=100/100; TQAmean13.90,max30,over4=97/100. Each selected round<=2; response budget JSONnull, retirement/exhausted/binding0.
- **all_old_sources_preserved: PASS.** All213 source/prior-result SHA entries preserved; all23 frozen runtime SHA,6 frozen metric SHA and independently captured controller/helper hashes unchanged.
- **official_score_seals: PASS.** All18 dev/main score seals, aggregate/perquestion/provenance SHA verified;12 exact ordered whole-corpus reuses also match historical source seal/aggregate/perquestion SHA,6 new official scores. No historical runtime timing reuse.
- **spice_and_timing_separation: PASS.** Both new uncapped official SPICE runs use64G only resource change; input/output/log hashes valid, full IDs, no OOM/ignoredsentence/NPE. First new Java execution07:21:55.492UTC follows GPU-free stage07:21:54.938UTC. Orchestrator waits successful child exits and noGPU pids; resource events empty.
- **independent_numerical_recalculation: PASS.** Independently reconstructed draws from seeds and verified draw SHA.1065 numerical comparisons pass: all9 summaryrows timing/token/MAT/memory/budget/quality point values and95percent CIs,126 comparisonrows including approximate p andHolm, plus12 one-sided NI conditions. No official score or generation rerun.
- **output_retention: PASS.** All286 changed main output pairs retained:89ScienceQA,100COCO,97TextVQA. All raw failures/caps kept; all900 main generationcaps0. TextVQA uncapped has3new final-answer extraction failures, retained aszero-score observations.
- **result_interpretation: PASS.** Report distinguishes tps/MAT from request latency. Uncapped tps+3.94/+8.16/+7.63percent vs cap4; SQA/COCO request CI crosszero; TextVQA request+4.85percent, paired95percent difference[.0498,.1796]seconds withHolm p.0024. No universal latency improvement claimed.
- **quality_and_ni_scope: PASS.** SQA83->86percent; TQA72.3->73.3percent with3parsingfailures; COCOSPICE13.574->13.093percent,CHAIRs22->20percent. Only SQA uncapped-vs-cap4 fixed-pilot NIproxy passes; uncappedvsStrict NI notestablished for anytask, COCO/TQA vs cap4 notestablished. Human image-conditioned factual/coverage auditpending.
- **statistical_limits: PASS.** 5000 paired physicalimage/groupbootstrap, onepass clocks, Holm within task/metric3pairs only, no globalfamily claim. Prior data exposure and fixedDF CIDEr bootstrap disclosed. p>.05 not equivalence; NIproxy not overall quality guarantee; rare unseen harm cannot be excluded.
- **final_artifact_seals: PASS.** All19 final artifact SHA and18 officialseal SHA in FINAL_DELIVERY verified.9 initial delivery entries verified with one intentional reportSHA superseded by append-only provenance; stripping addendum exactly reconstructs original reportSHA.
- **pdf_and_report: PASS.** Performance PDF3pages andqualityPDF2pages readable with extractable task names/units. Report228lines and table values agree with numerical checks; all positive/negative quality changes and TextVQA latency/format degradation disclosed.

## Scientific interpretation

The single-factor ablation supports increased token throughput and MAT in this fixed pilot. It does not establish faster request completion for all tasks: TextVQA requests are4.85percent slower, while SQA/COCO request differences remain uncertain. Only the finite-pilot ScienceQA uncapped-versus-cap4 automatic NIproxy passes; uncapped versus Strict does not establish NI for anytask.

## Retained limitations

- Eachmethod measured once; image-bootstrap CI excludes repeated-run timing noise.
- Current test sets were previously exposed and are not independent unseen confirmation.
- ScienceQA uncappedvs-cap4 NIproxy pass applies only to this finite pilot under predeclared2pp tolerance and bootstrap assumptions, not a guarantee for unseen inputs or explanation correctness.
- COCO object hallucination/SPICE changes are mixed automatic evidence; factual precision and salient coverage require independent image-based human review.
- TextVQA improved answer-score point estimate coexists with statistically slower requests and3new format-extraction failures.
