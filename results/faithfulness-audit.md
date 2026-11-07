# Faithfulness audit: paper summaries vs. source text (test split, n=20)

**Method.** An AI agent (Claude) read the 20 test papers in `data/pairs/test.jsonl` in one manual pass. For each paper it compared every factual claim in the trained summary (`results/sft-dpo-v4/samples/`) and the base summary (`results/base/`) against `data/text/<id>.txt`. That file holds the title, abstract and introduction, which is the model's only input. The claims checked were numbers, method descriptions, results, comparisons, author credit and "why it matters" significance claims. Each problem is tagged as **wrong** (contradicts the source), **unsupported** (not stated in or directly implied by the source) or **overstated** (stronger than the source, e.g. a dropped hedge or caveat). Each also carries a severity (minor / moderate / major) as a judgment aid. Style was ignored, and reasonable paraphrase did not count as a problem. Every quoted summary phrase was checked by script to appear verbatim in its summary file. Full findings, with source evidence for each, are in `faithfulness-audit.json`.

**Limits.** This was a single pass by one AI reviewer, with no second annotator and no human adjudication. The minor/moderate line and the strictness about dropped hedges ("often", "to our knowledge", "suggest") are judgment calls, and another reviewer would land on somewhat different counts. Most minor items are dropped qualifiers, and a reasonable editor might accept them. Omissions (leaving out a caveat entirely, rather than contradicting it) were not counted. Only the input text was used as ground truth, so a claim that is true of the full paper but absent from the input counts as unsupported. Base summaries are longer on average, which gives them more room for errors.

## Per-paper problem counts (T = trained sft-dpo-v4, B = base-forced)

| Paper | T wrong | T unsupp. | T overst. | **T total** | B wrong | B unsupp. | B overst. | **B total** |
|---|---|---|---|---|---|---|---|---|
| 2606.19544 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 2606.24775 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 2 |
| 2607.05775 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
| 2607.16621 | 1 | 1 | 1 | 3 | 0 | 0 | 0 | 0 |
| 2607.19322 | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| 2607.20709 | 0 | 0 | 2 | 2 | 0 | 0 | 1 | 1 |
| 2607.27687 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 2608.07222 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| 2608.11242 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| 2608.25593 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| 2608.26701 | 0 | 0 | 1 | 1 | 1 | 0 | 2 | 3 |
| 2609.00546 | 1 | 0 | 1 | 2 | 0 | 0 | 1 | 1 |
| 2609.01588 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| 2609.16487 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2609.35738 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 2 |
| 2610.00972 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| 2610.09484 | 1 | 0 | 1 | 2 | 0 | 0 | 0 | 0 |
| 2610.32318 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 3 |
| 2610.35959 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 2 |
| 2610.36449 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
| **Total** | 5 | 2 | 13 | **20** | 4 | 6 | 12 | **22** |

## Totals

| | Trained | Base |
|---|---|---|
| Summaries with zero problems | 5/20 | 5/20 |
| Wrong | 5 | 4 |
| Unsupported | 2 | 6 |
| Overstated | 13 | 12 |
| **All problems** | **20** | **22** |
| Major / moderate / minor | 1 / 7 / 12 | 0 / 4 / 18 |

The two models make about the same number of problems overall. The trained model has fewer minor slips but more serious ones: one major and seven moderate problems, against none and four for base. The trained model's most common failure is dropping a hedge, and it shows up most in the "Why it matters" bullet. Several trained errors reuse a number that really is in the source but attach it to the wrong thing, so a digit-matching checker passes them.

## The 3 most serious trained-model problems

1. **2610.00972 (VeriHarness), wrong, major.** Summary: "the most frequent answer is wrong 47% of the time". Source (Fig. 2): "Most frequent answer correct 47%". The meaning is inverted: the most frequent answer is correct 47% of the time and wrong 53%.
2. **2607.27687 (Rehearse), wrong, moderate.** Summary: "On 366 same-baseline pairs". Source: "296 strictly same-baseline pairs ... with an auxiliary 70-pair worked-versus-worked ranking probe". The 366-pair benchmark is not all same-baseline.
3. **2610.09484 (MIMESIS), wrong, moderate.** Summary: "so agents trained on them do not match real user success rates". Source (Fig. 2): the mismatch is the "success rate of a fixed gpt-5.5 agent ... when each simulator plays the user role". It is an evaluation effect, not a training one. The same summary's "Why it matters" then turns this into an untested causal claim ("training trajectories that match real user difficulty, improving generalization").

**Implication for the article.** The claim that "every claim and number comes from the input" does not hold for the trained model. Only 5 of 20 trained summaries are fully clean. Five contain a factual error, and one of those reverses a statistic.
