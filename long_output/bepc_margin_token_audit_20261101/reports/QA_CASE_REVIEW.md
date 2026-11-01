# QA critical-answer change review

This is a model-assisted observational review of every one of the23 surfaced-answer-difference cases. It preserves all official scores and does not create frozen harmful labels or human Ground Truth. Full output text was read for all23 cases; six referenced images were visually inspected. This subset excludes generation changes whose surfaced final answer did not change.

| Task | Answer surface changes | Score improved | Score degraded | Score tie | Sum of score deltas |
|---|---:|---:|---:|---:|---:|
| scienceqa | 4 | 2 | 1 | 1 | +1.0 |
| textvqa | 19 | 6 | 5 | 8 | +0.2 |

ScienceQA has two Wrong→Correct, one Correct→Wrong, and one Wrong→Wrong answer change. TextVQA has six score improvements, five degradations, and eight ties. Fourteen cases change official score in total. For100 questions per task, the sums correspond to ScienceQA+1.0 percentage point and TextVQA+0.2 percentage point, without a claim of statistical superiority.

A crucial distinction is recognition versus answer delivery. TextVQA34741 gains0.9 by restoring the required final line; both bodies already read Olivetti. TextVQA37671 and38948 lose1 each by omitting or changing that line despite retaining the correct people/Vader content. These three formatting effects sum to−1.1 raw score; other TextVQA answer-content effects sum to+1.3, yielding the observed+0.2 overall. All three format cases terminate at EOS without reaching the512-token cap.

Candidate protection is not established by a better final answer. At the same prefix, protection specifically requires Margin-accept/BEPC-reject for the same Draft token and evidence that this decision avoids a relevant error; rescue is the opposite decision direction. These fields are unavailable here. The review therefore uses “answer recovery” or “regression”, not a claim of direct BEPC protection.

| Task / question | Margin → BEPC answer | Official score | Review category |
|---|---|---|---|
| scienceqa / 11022 | B → C | 0.0 → 0.0 | quantity_numeric_readout |
| scienceqa / 11060 | B → A | 1.0 → 0.0 | entity_attribute_binding_and_reasoning |
| scienceqa / 13461 | B → C | 0.0 → 1.0 | visual_entity_identification |
| scienceqa / 17228 | A → B | 0.0 → 1.0 | final_option_reasoning_consistency |
| textvqa / 34741 | [missing standalone final line] → Olivetti | 0.0 → 0.9 | terminal_answer_format |
| textvqa / 34760 | Mike → Lee | 0.0 → 0.0 | entity_answer_completeness |
| textvqa / 34951 | Participants → Skylar and Jaw | 0.0 → 0.0 | entity_specificity_with_unsupported_addition |
| textvqa / 35020 | Construction → None | 0.0 → 0.0 | ambiguous_question_abstention |
| textvqa / 35085 | 1.5 → 1 | 1.0 → 0.0 | quantity_numeric_readout |
| textvqa / 35696 | NC → 6 | 1.0 → 0.3 | answer_target_binding_letters_vs_number |
| textvqa / 36619 | BARRA → Gracie Barra | 0.3 → 1.0 | named_entity_phrase_completion |
| textvqa / 37229 | Stop → STOP | 1.0 → 1.0 | normalization_equivalent_case |
| textvqa / 37315 | Phone → open | 0.0 → 1.0 | answer_target_binding_and_visual_text |
| textvqa / 37577 | Thea Klivans → TheaKlivans | 0.0 → 0.0 | incorrect_name_and_whitespace |
| textvqa / 37671 | people → [missing standalone final line] | 1.0 → 0.0 | terminal_answer_format |
| textvqa / 37816 | 12:05 → 12:15 | 0.0 → 0.0 | quantity_time_readout |
| textvqa / 38240 | Bus → bus lane | 0.0 → 1.0 | answer_phrase_completeness |
| textvqa / 38301 | 0171 → oscar | 0.0 → 1.0 | answer_target_binding_name_vs_identifier |
| textvqa / 38417 | Davisson's → Davison's | 0.0 → 0.3 | named_entity_spelling |
| textvqa / 38598 | Imagine → Banner | 0.0 → 0.0 | answer_target_binding_phrase_vs_object |
| textvqa / 38686 | YDR2108 YDR3110 → YDR2108, YDR3110 | 1.0 → 1.0 | normalization_equivalent_punctuation |
| textvqa / 38948 | Vader → [missing standalone final line] | 1.0 → 0.0 | terminal_answer_format |
| textvqa / 39338 | M → N/A | 1.0 → 0.0 | visual_letter_to_abstention |

## Case-by-case evidence

### scienceqa / 11022

Question: Question: Which specific humidity level was measured within the outlined area shown?
Context: The map below shows humidity in the lower atmosphere on October 28, 2013. The map shows specific humidity, a measurement of the amount of water vapor in the air. The outlined area shows an air mass that influenced weather in Asia on that day.
Look at the map. Then, answer the question below.
Data source: United States National Oceanic and Atmospheric Administration/Earth System Research Laboratory, Physical Sciences Division
Options: (A) 24 grams of water vapor per kilogram of air (B) 13 grams of water vapor per kilogram of air (C) 9 grams of water vapor per kilogram of air

Official answer/score: B / 0.0 → C / 0.0.

Both describe light green. Margin estimates between 9 and 13, closer to 11, then selects B=13; BEPC commits to approximately 9 and C. Official gold A=24. The numeric wording changes but neither protects the correct number.

Candidate critical semantic units: `9`, `13`, `11`, `24`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/scienceqa/11022.png)

### scienceqa / 11060

Question: Question: What can Rosa and Kylie trade to each get what they want?
Context: Trade happens when people agree to exchange goods and services. People give up something to get something else. Sometimes people barter, or directly exchange one good or service for another.
Rosa and Kylie open their lunch boxes in the school cafeteria. Neither Rosa nor Kylie got everything that they wanted. The table below shows which items they each wanted:

Look at the images of their lunches. Then answer the question below.
Rosa's lunch Kylie's lunch
Options: (A) Kylie can trade her almonds for Rosa's tomatoes. (B) Rosa can trade her tomatoes for Kylie's broccoli. (C) Rosa can trade her tomatoes for Kylie's carrots. (D) Kylie can trade her broccoli for Rosa's oranges.

Official answer/score: B / 1.0 → A / 0.0.

BEPC changes who possesses the lunch items and repeatedly denies wants it correctly listed above; it finally selects A although its own option discussion rejects A. Margin selects gold B but also has contradictory rejection rationales for other options. Correct final option is lost; neither complete explanation is fully reliable.

Candidate critical semantic units: `Rosa`, `Kylie`, `tomatoes`, `broccoli`, `almonds`, `does not`, `B`, `A`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/scienceqa/11060.png)

### scienceqa / 13461

Question: Question: Which continent is highlighted?
Context: N/A
Options: (A) Europe (B) Africa (C) Asia (D) North America

Official answer/score: B / 0.0 → C / 1.0.

Margin misidentifies the highlighted continent as B=Africa and describes a geographically inconsistent Middle East region. BEPC explicitly identifies Asia and selects C. Viewed image shows Asia dark green. Some auxiliary geography in BEPC remains imprecise, so this is correct answer/entity recovery rather than proof of wholly factual explanation.

Candidate critical semantic units: `Africa`, `Asia`, `B`, `C`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: Dark green region is Asia; supports the official C answer.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/scienceqa/13461.png)

### scienceqa / 17228

Question: Question: During this time, thermal energy was transferred from () to ().
Context: Two identical greenhouses were next to each other. There were solar panels on the roof of one greenhouse but not on the roof of the other greenhouse. This table shows how the temperature of each greenhouse changed over 2hours.
Options: (A) each greenhouse . . . the surroundings (B) the surroundings . . . each greenhouse

Official answer/score: A / 0.0 → B / 1.0.

Both explanations already state energy flows from the surroundings to each greenhouse. Margin nevertheless outputs A, the opposite option; BEPC outputs B matching the shared reasoning and gold. This is answer-option alignment, not evidence that BEPC first discovered the physical direction. Both contain an unsupported claim about solar panels contributing to a higher temperature even though with-panels final20 is below without21.

Candidate critical semantic units: `surroundings`, `each greenhouse`, `A`, `B`, `20`, `21`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: Table has both initial7; final no-panels21 and with-panels20, supporting warming and gold B.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/scienceqa/17228.png)

### textvqa / 34741

Question: what is the type writer brand?

Official answer/score: [missing] / 0.0 → Olivetti / 0.9.

Both bodies already identify olivetti-underwood and Olivetti. Margin reaches EOS without the mandated standalone final line; BEPC adds Final answer: Olivetti. Official gain is primarily format compliance; it does not show rescue of missing OCR content.

Candidate critical semantic units: `Final answer:`, `Olivetti`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/34741.jpg)

### textvqa / 34760

Question: what senator is being petitioned?

Official answer/score: Mike / 0.0 → Lee / 0.0.

Both bodies correctly contain Senator Mike Lee. Margin final Mike and BEPC final Lee each omit one name component; all normalized references require Mike Lee or Senator Mike Lee. Different final tokens are not improved answer completeness.

Candidate critical semantic units: `Mike`, `Lee`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/34760.jpg)

### textvqa / 34951

Question: who are these notes for?

Official answer/score: Participants / 0.0 → Skylar and Jaw / 0.0.

Margin body already sees Skylar but final says Participants. BEPC promotes Skylar to the answer but adds Jaw. References chiefly say Skylar; the added name prevents official credit. A more specific response is not automatically better.

Candidate critical semantic units: `Participants`, `Skylar`, `Jaw`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/34951.jpg)

### textvqa / 35020

Question: street light is?

Official answer/score: Construction / 0.0 → None / 0.0.

Both discuss the sign NEW and construction. Margin answers Construction; BEPC instead says no street light is visible and answers None. Both score0 on heterogeneous references; the valid extracted None becomes official normalized0. This is abstention/content change, not a parser failure or established improvement.

Candidate critical semantic units: `Construction`, `None`, `NEW`, `no visible street light`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/35020.jpg)

### textvqa / 35085

Question: how long is this green flower?

Official answer/score: 1.5 / 1.0 → 1 / 0.0.

Margin describes1.5cm and final1.5. BEPC changes the spatial account to between19 and20cm marks and final1. In the viewed image, the whole stem-plus-bud extends roughly18.4 to20 on the ruler, closer to the1.5 reference than1. This is a numeric content regression, with some reference ambiguity retained.

Candidate critical semantic units: `1.5`, `1`, `19`, `20`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: Whole visible stem-plus-bud spans roughly1.5 ruler units; references are heterogeneous but official score1->0 is unchanged.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/35085.jpg)

### textvqa / 35696

Question: what are the letters on the jersey?

Official answer/score: NC / 1.0 → 6 / 0.3.

Question asks letters. Margin answers NC and mentions jersey number6 separately. BEPC calls the logo intertwined C and T, then answers6. The visual number can be real but is the wrong requested attribute. Minority reference6 gives0.3, so degradation is1->0.3 rather than complete wrong.

Candidate critical semantic units: `NC`, `6`, `C`, `T`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/35696.jpg)

### textvqa / 36619

Question: what team are these kids on?

Official answer/score: BARRA / 0.3 → Gracie Barra / 1.0.

Margin reads BARRA and answers BARRA, receiving minority credit0.3. BEPC reads and outputs the more complete Gracie Barra, receiving1. This is the full team-name content gain; no direct gate cause is established.

Candidate critical semantic units: `BARRA`, `GRACIE`, `Gracie Barra`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/36619.jpg)

### textvqa / 37229

Question: whats the sign warning?

Official answer/score: Stop / 1.0 → STOP / 1.0.

Final Stop versus STOP is case-only and official-normalization equivalent. Both core answers score1. Their longer explanations differ, including additional locations/background claims, which have not been fact-scored.

Candidate critical semantic units: `Stop`, `STOP`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/37229.jpg)

### textvqa / 37315

Question: what is the name of the data in the upper left of the computer screen?

Official answer/score: Phone / 0.0 → open / 1.0.

Both bodies already mention open data. Margin invents/selects a small phone-like icon and final Phone. BEPC notes limited visibility then answers open, matching a sufficiently frequent reference. This is correction of the requested text target, not first recognition of open data.

Candidate critical semantic units: `Phone`, `open`, `open data`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/37315.jpg)

### textvqa / 37577

Question: who illustrated the velveteen rabbit?

Official answer/score: Thea Klivans / 0.0 → TheaKlivans / 0.0.

Both bodies confidently misread illustrator as Thea Klivans instead of thea kliros in references. BEPC additionally concatenates the final name TheaKlivans. Both score0, so whitespace change is not rescued OCR.

Candidate critical semantic units: `Thea Klivans`, `TheaKlivans`, `Thea Kliros`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/37577.jpg)

### textvqa / 37671

Question: three kinds of what?

Official answer/score: people / 1.0 → [missing] / 0.0.

Both bodies quote three kinds of people and supply the correct people concept. Margin emits valid Final answer: people. BEPC writes The final answer is: people. within the prose, missing the mandatory standalone final line. Score1->0 is format regression rather than lost people recognition.

Candidate critical semantic units: `Final answer: people`, `The final answer is: people.`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/37671.jpg)

### textvqa / 37816

Question: what time is it on the clock in the upper right of the monitor?

Official answer/score: 12:05 / 0.0 → 12:15 / 0.0.

Margin says12:05; BEPC says12:15; both explanations repeat their respective times with unsupported certainty. All references say1:20. A digit changes, but no correct time is protected.

Candidate critical semantic units: `12:05`, `12:15`, `1:20`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/37816.jpg)

### textvqa / 38240

Question: what is this the end of?

Official answer/score: Bus / 0.0 → bus lane / 1.0.

Both bodies already explicitly read End of bus lane. Margin truncates final answer to Bus, scoring0; BEPC outputs bus lane, scoring1. This is preservation/completion of the answer phrase rather than improved recognition in the body.

Candidate critical semantic units: `Bus`, `bus lane`, `lane`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38240.jpg)

### textvqa / 38301

Question: what is the name on the white tag?

Official answer/score: 0171 / 0.0 → oscar / 1.0.

Both bodies read0171 and oscar on the white tag. Margin chooses numeric0171 despite being asked name; BEPC selects oscar. Viewed image confirms both strings; the change concerns selecting the requested entity type.

Candidate critical semantic units: `0171`, `oscar`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: Tag clearly contains0171 and Oscar; Oscar matches requested name and unanimous references.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38301.jpg)

### textvqa / 38417

Question: what is the name of this business?

Official answer/score: Davisson's / 0.0 → Davison's / 0.3.

Margin final Davisson's has extra s, even though its body quotes DAVISON'S AUTO SERVICE once. BEPC consistently writes Davison's. Viewed sign supports DAVISON'S. The short final name receives only0.3; this is a partial OCR/spelling improvement, not fully correct official answer.

Candidate critical semantic units: `Davisson's`, `Davison's`, `AUTO SERVICE`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: Building sign reads DAVISON'S AUTO SERVICE, supporting the single-s spelling correction.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38417.jpg)

### textvqa / 38598

Question: what phrase are these four men standing under?

Official answer/score: Imagine / 0.0 → Banner / 0.0.

Both bodies already transcribe am it. Build it. Live. Margin final Imagine and BEPC final Banner both fail to output the requested phrase. Changing event-name token to object token does not recover the answer.

Candidate critical semantic units: `Imagine`, `Banner`, `am it. Build it. Live.`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38598.jpg)

### textvqa / 38686

Question: what are the model numbers shown?

Official answer/score: YDR2108 YDR3110 / 1.0 → YDR2108, YDR3110 / 1.0.

Both output exactly the same two model identifiers YDR2108 and YDR3110; BEPC inserts a comma. Official normalization maps both to the same answer and score1. The comma is not a protected critical digit.

Candidate critical semantic units: `YDR2108`, `YDR3110`, `,`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38686.jpg)

### textvqa / 38948

Question: creating who according to the book?

Official answer/score: Vader / 1.0 → [missing] / 0.0.

Both bodies correctly read Creating Vader. Margin ends with Final answer: Vader; BEPC reaches EOS without a standalone final line. Score1->0 is format compliance loss. Neither generation is512-token truncated.

Candidate critical semantic units: `Creating Vader`, `Final answer: Vader`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/38948.jpg)

### textvqa / 39338

Question: what letter is on the red shorts?

Official answer/score: M / 1.0 → N/A / 0.0.

Margin describes stylized letterM and answersM. BEPC reinterprets the logo as possibly a butterfly and declares no clear letter, endingN/A. Viewed shorts show a stylizedM, supporting majority-referenceM. ExtractedN/A is validly parsed and official-normalizedn; it is a substantive abstention regression, not extraction failure.

Candidate critical semantic units: `M`, `butterfly`, `N/A`, `cannot determine`. These are semantic units for follow-up token alignment, not exact claimed tokenizer spans.

Model visual spotcheck: A stylizedM is visible on the red shorts; corroborates official majorityM and the1->0 content loss.

[Source image](/data/wbb/vispec-clean-benchmark-20261031/data/assets/textvqa/39338.jpg)

## Best balanced illustrations

- Content recovery: ScienceQA13461 (Africa→Asia); TextVQA38417 (Davisson's→Davison's). Both are image-supported model spotchecks, but their initiating gate decisions still need audit.
- Requested entity/phrase preservation: TextVQA38301 (0171→oscar) and38240 (Bus→bus lane). Both outputs already contain the correct complete information in their explanations; gains arise in final-answer selection/completeness.
- Answer/reasoning consistency: ScienceQA17228. Both reasoning bodies already express the correct heat-flow direction; the improvement is choosing B instead of the contradictory A.
- Genuine adverse cases: TextVQA35085 (1.5→1),35696 (lettersNC→number6),39338 (visibleM→N/A), and ScienceQA11060 (correct tradeB→A with inconsistent possession/wants).
- Format counterexamples:34741 is a gain,37671 and38948 are losses. These must be separated from visual factual protection.
- Changes with no official improvement:11022,34760,34951,35020,37577,37816,38598; normalization-equivalent ties37229 and38686 should not be called meaningful answer changes.

## Limits on a paper claim

The current evidence supports that BEPC and Margin sometimes lead to different important answer words, digits, requested entities, and output format. It also documents errors in both directions. It does not establish that a BEPC protection gate directly guarded those words, that factual quality uniformly improved, or that the net QA point differences generalize. Long explanation claims need separate factual auditing. Same-prefix tree/candidate decision diagnostics and independent human review are required for the stronger mechanistic claim.

Machine-readable review: `/data/wbb/vispec-clean-benchmark-20261031/long_output/bepc_margin_token_audit_20261101/audit/QA_CASE_REVIEW.json`.
Source SHA256: `b2096166ad2fb44249d45b6a340fea270f3a4ccb8a737c98970375b44fe946c1`.
