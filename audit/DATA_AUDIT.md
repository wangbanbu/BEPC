# Data preparation audit

The frozen benchmark contains 20 development and 100 test image groups for each of MME, GQA, TextVQA, ScienceQA, and COCO. Each method runs three repetitions on both splits. MME retains both official yes/no questions for every sampled image group. Across all tasks there are 600 distinct RGB images, 720 question rows, and no development/test image overlap, including across task boundaries.

The selected cohort and model inputs remain frozen. Historical-image overlap discovered after the initial seal is disclosed through an additive reporting erratum; no images were replaced and no inference was repeated to address it.

| Task | Dev groups | Test groups | Dev / test question rows | Verified historical-overlap groups, dev / test |
|---|---:|---:|---:|---:|
| mme | 20 | 100 | 40 / 200 | 20 / 100 |
| gqa | 20 | 100 | 20 / 100 | 0 / 0 |
| textvqa | 20 | 100 | 20 / 100 | 1 / 3 |
| scienceqa | 20 | 100 | 20 / 100 | 0 / 0 |
| coco | 20 | 100 | 20 / 100 | 1 / 2 |

## Sources and deterministic selection

The selection seed is `2026103109`. Eligible questions are ordered by SHA256 of the seed, task, and question identity; one group per image is retained. MME first allocates category quotas, then uses the same fixed hash ordering with global RGB deduplication across all categories and both splits. All fourteen categories appear in development (one or two groups/category) and test (seven or eight/category). The test allocation gives eight groups to color and scene and seven to each remaining category.

- GQA: labelled official `val_balanced_questions.json` and local official images; prior question IDs were also mapped through available official testdev/val/test metadata to image IDs.
- TextVQA: official version0.5.1 validation questions and ten human answers. The local author ZIP is incomplete, so complete local entries are recovered without rewriting it; other selected members are fetched by bounded HTTP ranges from the canonical author archive. Header offsets, ZIP compression method, uncompressed size, CRC32, PIL decoding, and file/RGB hashes are checked. Selected images comprise19 remote+1 local development images and97 remote+3 local test images. The member index SHA256 is `0e9f9b2fcd1e8932196d23f58a2906e8ae06c5f2239225acba588be463b3cbbc`.
- ScienceQA: image-containing official test questions. Every local test-parquet row is checked against official `pid_splits.json` and `problems.json` for question, choices, and answer before mapping row index to official problem ID. The selected120 official answer/choice joins are checked again independently. The100 recorded `sqa:test-row-N` image aliases were mapped through this official split; none matches a selected ScienceQA question.
- COCO: official Karpathy test metadata, five reference sentences per image, and local validation image parquets. Actual COCO IDs are read from `COCO_val2014_<id>.jpg`, never inferred from the unrelated parquet numeric `id` column.
- MME: all1187 official complete two-question groups are historically present in the old project, so project-unseen MME sampling is impossible. Historical reuse was explicitly approved for this benchmark. Selection uses image content and category, with no old performance values.

The initial candidate counts after the frozen historical exclusion were GQA105743 questions/8317 images, TextVQA4308 questions/2754 images when the canonical remote archive index is included, ScienceQA717 image questions, COCO3612 images, and MME0 fresh groups. MME was therefore sampled from the full official1187-group pool with an explicit historical-reuse flag.

## Historical audit and errata

`DATA_PRIOR_IDENTITIES_FAST.json` records20929 identity-bearing source files, each with SHA256, from `/data/wbb/vispec-official-reproduction` and `/data/wbb/ViSpec`. Thirty-four local event/feature parquets contribute identity columns only through `DATA_PARQUET_IDENTITY_SUPPLEMENT.json`. A visibility check finds360 additional ignored paths;68 contain eligible identities. Their additive supplement finds no additional exact selected-ID/hash match. Namespaced image IDs are then checked separately, rather than treating arbitrary equal numeric IDs across datasets as the same image.

Seven selected non-MME images have verified prior task-prefixed identities: TextVQA dev38921 and test38948/37315/39465; COCO dev185487 and test199403/49984. COCO49984 additionally has an exact historical byte SHA256. Each case has an original per-request source path, source SHA256, line number, and prior identity in `DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json`. Their frozen runtime `fresh_to_project=true` fields are erroneous; the additive supplement changes their effective reporting flag to false. The runtime files themselves are unchanged. Effective no-recorded-overlap counts are GQA20/100, ScienceQA20/100, TextVQA19/97, COCO19/98, and MME0/0, development/test respectively.

The original canonicalizer did not strip task-prefixed identity namespaces, and COCO bytes were checked only after selection/materialization. Those limitations are retained as provenance findings. The benchmark supports within-run held-out development/test evaluation. It must not be called wholly project-unseen, and absence from this two-root identity audit does not establish absence from external projects or model pretraining.

Only identity fields and generation-field presence determine historical eligibility. Old quality, timing, generated-caption quality, or desired hypotheses do not determine the selected cohort. Hashing whole source files supplies provenance; it does not select by those files' performance values.

## Inputs, labels, and validation

Runtime files are `data/{mme,gqa,textvqa,scienceqa,coco}/{dev,test}.json`. Gold maps are separate `dev_labels.json` and `test_labels.json`, keyed by string `question_id`. All720 selected labels are rejoined to official sources; all image byte/RGB hashes, labels, ordinals, group sizes, file SHA256 values, and absence of gold fields in runtime rows pass. The group count is600, with no image duplicated across tasks or splits; MME alone repeats its image within its two-question group.

System prompt: `You are a helpful assistant.` GQA/TextVQA use the original question plus `Answer the question using a single word or phrase.` MME keeps exactly one `Please answer yes or no.` ScienceQA uses official question, hint/context, and lettered choices, then `Answer with the option letter only.` Captions, lectures, solutions, and few-shot examples are excluded from ScienceQA inputs. COCO uses `Describe this image in one concise sentence.` All methods use128 maximum new tokens. Processor/model/tree settings are frozen by the parent experiment configuration; the common Qwen visual budget differs from older COCO-only timing runs, so those old timings are not directly comparable.

Primary artifacts:

- `DATA_MANIFEST.json`: frozen file paths, counts, and hashes.
- `DATA_VALIDATION.json`: all720 source-label joins, hashes, group/disjointness checks.
- `DATA_HISTORICAL_OVERLAP_SUPPLEMENT.json`: canonical effective provenance flags and verified historical evidence.
- `DATA_NAMESPACE_ALIAS_CHECK.json`, `DATA_SCIENCEQA_ROW_NAMESPACE_CHECK.json`: task-prefix and official-row mappings.
- `DATA_IGNORED_IDENTITY_SUPPLEMENT.json`: additional visibility audit.
- `DATA_MME_PRIOR_SOURCES.json`: historical MME coverage evidence.
- `DATA_AUDIT_EXTRACTION_AMENDMENT.json`: slow initial extraction stopped before sampling and replaced by native identity extraction.

The data-integrity gate passes. The stronger claim that every selected image is unseen in project history fails and is not required for the actual paired decoding benchmark. No resampling, outcome access, model generation, or parameter adjustment is performed by this data-preparation task.
