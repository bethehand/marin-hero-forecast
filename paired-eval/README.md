# Paired comparisons for Marin's eval policy

This is an independent analysis, not affiliated with Marin or Open Athena. It uses only public data:
- the per-sample outputs Marin published in [marin-community/grug-67b-a2b-snowball-post-sft-evals](https://huggingface.co/datasets/marin-community/grug-67b-a2b-snowball-post-sft-evals), at revision `ad03b3669e`;
- per-instance SWE-bench Verified results from [SWE-bench/experiments](https://github.com/SWE-bench/experiments), at commit `40f164d5b8`;
- the tournament artifacts Marin published:
  - Snowball v0 ([#10199](https://github.com/marin-community/marin/issues/10199), gist revision `69708d2f38`);
  - Snowball v0.1 ([#10283](https://github.com/marin-community/marin/issues/10283), gist revision `191228e26b`).

**Context.** Marin's eval policy ([#10197](https://github.com/marin-community/marin/issues/10197)) calls a benchmark comparison decisive only when `abs(d) > 1.96 * hypot(s_a, s_b)`, with each task-level SEM multiplied by 1/√0.9. The tournament code ([`pairwise.py`](https://github.com/marin-community/marin/blob/6885cabd456787794551a2e41fbff99188fedd37/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/analysis/snowball_tournament/pairwise.py)) applies the same rule, which treats the two models' scores as independent. Both models answer the same tasks, so the relevant uncertainty is the SE of the per-task difference. With one sample per task, that SE is smaller whenever per-task outcomes are positively correlated.

## Findings

1. **In Marin's own per-sample outputs, pairing cuts the SE on every benchmark checked.**
   - The models are Marin's Grug-67B-A2B SFT checkpoint (step 630) and six MoE models from other families: DeepSeek-V2-Lite-Chat, Ling-lite, Moonlight-16B-A3B, Qwen3-30B-A3B-Thinking-2507, Qwen3.5-35B-A3B and Qwen3-Next-80B-A3B-Thinking.
   - The table counts pairs whose accuracies differ by at most 15 points. On the six single-sample benchmarks, the median cut is 6–37%.

   | Benchmark | Tasks | Pairs within 15 points | Paired / unpaired SE, median | Range |
   |---|---:|---:|---:|---|
   | ARC-Challenge (25-shot) | 1,172 | 21 | 0.63 | 0.50–0.69 |
   | BoolQ | 3,270 | 21 | 0.77 | 0.71–0.82 |
   | WinoGrande (5-shot) | 1,267 | 21 | 0.78 | 0.68–0.86 |
   | MATH-500 | 500 | 7 | 0.74 | 0.42–0.92 |
   | GSM8K (flexible-extract) | 1,319 | 7 | 0.94 | 0.82–0.96 |
   | GPQA Diamond | 198 | 6 | 0.83 | 0.66–1.04 |
   | AIME24 (3 samples per problem, averaged per problem) | 30 | 7 | 0.60 | 0.49–1.03 |

   - How scores were read:
     - GSM8K uses the second row per question (flexible-extract). It matches the published flexible-extract scores exactly.
     - In this export, GPQA Diamond and AIME24 have `correct` set to true on every row. They are rescored from `extracted` against the reference answer. The export keeps the first repetition of each run, and the rescored counts equal the published first-repetition counts in all 7 GPQA Diamond runs and all 21 AIME24 runs.
     - ARC-Challenge's `correct` is `acc_norm`, matching the published acc_norm exactly. Scoring by `acc` instead gives a median of 0.61 (0.51–0.71).
     - HumanEval+, MBPP+ and IFEval rows also have `correct` set to true and carry no metrics. Rescoring them would mean running the code or re-checking the instructions, so they are not included.
   - No two of these models share a base model, though three are Qwen models. On six of the seven benchmarks, the lowest ratio is a Qwen–Qwen pair; the exception is GSM8K. Marin's tournament candidates do share a base, and finding 2 suggests same-family pairs gain more.

2. **On SWE-bench Verified, pairing cuts the SE by about 30% for systems of similar strength, and slightly more within one model family.**
   - 135 systems have per-instance results. 467 pairs of them, drawn from 40 systems, both solve 75–225 of the 500 tasks and are at most 50 tasks apart. For these pairs, the paired/unpaired ratio has a median of 0.71; the 10th–90th percentile range is 0.66–0.76.
   - Under the policy rule on 500 tasks, the gap needed to call a winner falls from 6.3 to 4.4 points (medians).
   - The 10 pairs below share a base model: fine-tunes of one base, two versions or submissions of one model, or one model in different harnesses. Some differ in two of these ways.
   - Their ratios range from 0.24 to 0.72, with a median of 0.65. Nine of the ten are below 0.71.

   | Pair | Solve rates | Paired / unpaired SE |
   |---|---|---:|
   | Qwen2.5-Coder-32B-Instruct fine-tunes: SWE-agent-LM-32B vs Skywork-SWE-32B | 40.2% vs 38.0% | 0.66 |
   | Qwen3-Coder-30B-A3B-Instruct (OpenHands) vs an EntroPO fine-tune of it (R2E-Gym) | 51.6% vs 52.2% | 0.60 |
   | Lingma SWE-GPT 72B, two submissions | 25.0% vs 28.8% | 0.69 |
   | Lingma SWE-GPT 7B, two submissions | 10.2% vs 18.2% | 0.72 |
   | SWE-Fixer, two versions (they disagree on 13 of 500 tasks) | 30.2% vs 32.8% | 0.24 |
   | Devstral Small 2505 (OpenHands) vs 2507 (SWE-agent) | 46.8% vs 38.0% | 0.64 |
   | GLM-4.5 vs GLM-4.6, Z.ai's own submissions | 64.2% vs 68.2% | 0.56 |
   | Kimi K2: OpenHands (0711-preview) vs SWE-agent (kimi-k2-instruct) | 65.4% vs 53.4% | 0.68 |
   | Kimi K2: OpenHands (0711-preview) vs Lingxi (0905-preview) | 65.4% vs 71.2% | 0.54 |
   | Kimi K2: SWE-agent (kimi-k2-instruct) vs Lingxi (0905-preview) | 53.4% vs 71.2% | 0.68 |

3. **With several samples per task, the task has to be the unit.**
   - In the AIME24 outputs above (3 samples per problem), the task-level SE is 1.45–1.60× the SE that treats each sample as independent, for Grug and the three Qwen models.
     - Ling-lite and Moonlight solve the same problems in every sample, so they sit at the 3-sample ceiling of 1.75×.
     - The factor grows with the number of samples per task.
   - In the tournaments, most cells on these benchmarks use the per-sample count: 71 of 84 cells in v0 and 75 of 92 in v0.1. The benchmarks are TBLite and TB2.1 (3 trials per task) and AIME24 and OlympiadBench (10 samples per problem).
   - GPQA Diamond also averages 3 samples per question, but its cells count the 198 questions.
   - **AIME24 mixes the two bases.**
     - 13 of 21 cells in v0 and 15 of 23 in v0.1 use n = 30, and the rest n = 300 (AIME24 has 30 problems). In `tournament_cell_statistics.csv` their `sem_basis` is `binary_score_from_record_coverage` and `corrected_replay_trial_rewards`.
     - At p ≈ 0.7, that puts the threshold at about 25 points between two n = 30 cells and about 8 points between two n = 300 cells.
     - OlympiadBench mixes them too in v0.1, where 2 of 23 cells use n = 30. They are the two Smartie checkpoints, v0.1's pick and runner-up, which also use n = 30 on AIME24.
   - On these benchmarks, a task-level paired SE can come out larger or smaller than the current one. In these 3-sample AIME24 outputs, pairing (0.60×) roughly offsets the clustering factor. The tournament's n = 300 cells have 10 samples per problem, where the clustering factor is larger, so the offset need not carry over.

4. **In Marin's published tournaments, half the comparisons are ties, and many sit near the threshold.** Shares are of all comparisons.

   | Tournament | Candidates | Benchmark comparisons | Ties | Ties with \|d\| ≥ 0.71 × threshold | Ties with \|d\| ≥ 0.5 × threshold |
   |---|---:|---:|---:|---:|---:|
   | v0 ([#10199](https://github.com/marin-community/marin/issues/10199)) | 21 | 4,830 | 2,475 (51%) | 477 (10%) | 963 (20%) |
   | v0.1 ([#10283](https://github.com/marin-community/marin/issues/10283)) | 23 | 5,819 | 2,988 (51%) | 573 (10%) | 1,145 (20%) |

5. **How the selections respond if the thresholds are scaled by a factor r.**
   - This is a stand-in for pairing. The per-sample rewards behind the tournament cells are not public, so one r is applied to every comparison.
   - Scoring is exact and follows `pairwise.py`, including its tie-break: pooled win rate, then model ID. At r = 1 it reproduces the published win rates to within 1e-15.
   - **v0.1:** the pick, Smartie step14 (mean pairwise win rate 0.684, against 0.633 for the runner-up), holds for every r from 1.00 down to 0.50.
   - **v0:** the pick was Campaign3, ahead of Campaign2 by half a point in 460 benchmark comparisons (0.621 and 0.620). Head to head, the two tied on 17 of 23 benchmarks.
     - Between r = 0.98 and 0.96 the lead changes hands several times, with exact ties in between that the tie-break settles.
     - Campaign2 leads outright at every r ≤ 0.96, down to 0.50.
   - **Scaling only the other 19 benchmarks** gives the same picture:
     - Ties with |d| ≥ 0.71× and ≥ 0.5× the threshold are 9.5% and 19.5% (v0) and 9.5% and 19.2% (v0.1) of those comparisons.
     - v0.1's pick holds.
     - In v0, Campaign2 leads outright at every r ≤ 0.928. From 0.934 down to 0.929, the two are exactly tied.

## What a paired SE needs

For each pair of models and each benchmark:
- let d_t be A's mean reward on task t minus B's, over the tasks both models scored;
- with several samples per task, average a task's samples first, so that n counts tasks;
- the SE is sd(d_t)/√n.

The trial archives behind the tournament hold per-sample rewards, and its statistics code already reads them for continuous metrics.

## Caveats

- **Marin's outputs have no same-base pairs.** No two of the seven models share a base model; three are Qwen models with different bases. Only one of the 3 GPQA Diamond repeats, and one of the 10 AIME24 repeats per seed, is in the export.
- **SWE-bench is an outside reference.** Its systems differ from Marin's candidates in base model, scaffold and training. The 467 pairs come from 40 systems, so they are not independent samples.
- **One r for everything.** A single r ignores that the real paired SE varies by benchmark and by pair.
- **Task sets are not quite identical.** Scored sets differ after infrastructure exclusions: SWE-bench Verified cells score 451–500 tasks. MRCR's item set depends on the served context: one candidate in each tournament was scored on 602 items, the others on 1,196.
- **No per-sample data from the tournament cells was used.**

## Reproduce

Run these from a scratch directory outside this repo. `marin_outputs_paired.py` needs `pandas` and `pyarrow`.

```bash
# Marin's per-sample outputs (downloads about 76 MB at the pinned revision)
python <repo>/paired-eval/code/marin_outputs_paired.py cache <repo>/paired-eval/results/marin_outputs_paired.json

# SWE-bench Verified: check out only the results files at the pinned commit
git clone --filter=blob:none --no-checkout https://github.com/SWE-bench/experiments.git swe_exp
cd swe_exp && git sparse-checkout init --no-cone
git sparse-checkout set --no-cone '/evaluation/verified/*/results/results.json'
git checkout 40f164d5b8f1d249bf95a6df8b74b577fd8e519d && cd ..
python <repo>/paired-eval/code/swebench_paired.py swe_exp <repo>/paired-eval/results

# Marin's published tournament artifacts at the pinned revisions
git clone https://gist.github.com/penfever/8baa8321ad1d08ce514be90a25a5c0d2.git v0
git -C v0 checkout 69708d2f38b78389f8f7d20cbc504190f7e9095f
git clone https://gist.github.com/penfever/b7deea3a2eaec1642a56c0d67ba1e41a.git v01
git -C v01 checkout 191228e26bc3c2ef1a3c2496a7fd192d10f3c1d8
python <repo>/paired-eval/code/tournament_sensitivity.py v0 v01 <repo>/paired-eval/results/tournament_sensitivity.json
```

The outputs are in `results/`:
- `marin_outputs_paired.json`: per benchmark, every pair, and the AIME24 task-level vs per-sample SE;
- `swebench_summary.json`, `swebench_pairs.csv` (the 467 pairs) and `swebench_family_pairs.csv`;
- `tournament_sensitivity.json`: ties and near-threshold ties for both scalings, and every change of leader as r goes from 1.0000 down to 0.5000 in steps of 0.0001.
