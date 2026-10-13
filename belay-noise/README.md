# Belay seed noise: which Paloma subsets carry it

This is an independent analysis, not affiliated with Marin or Open Athena. It uses only the public W&B runs of Marin Belay v1 ([marin#9451](https://github.com/marin-community/marin/issues/9451)) in `marin-community/marin_moe`.

Belay keeps a change when its seed-paired d512 runs beat the current best Paloma macro by more than 0.005. The questions here are:
- How much of the noise in that comparison comes from each Paloma subset?
- Would a macro without `ptb` and `wikitext_103` rank configs more reproducibly?

These two subsets were chosen in advance, not by their noise in this data: they are the ones with copies in the hero run's training data ([marin#9874](https://github.com/marin-community/marin/issues/9874), [../eval-audit](../eval-audit)).

## Data
- Every finished `lc1-*` run, fetched 2026-10-13 03:17 UTC: 2,332 runs.
- **Seed pairs.** A seed pair is a seed-1 run named after its seed-0 partner with an `-s1` token, with identical config apart from `trainer.seed`.
  - Per-run output paths are ignored.
  - Both runs need the final eval of all 16 Paloma subsets and all 7 Uncheatable subsets.
  - At d512 this gives 404 pairs (batches 2–260 plus 14 pairs from unnumbered runs, which are grouped by launch day), launched 2026-09-25 to 10-08.
  - Each pair's two runs were launched within 1.3 h of each other.
- **What a seed changes.** `data_seed` is unset in every pair, so `trainer.seed` sets both the initialization and the data order: fast_track's `train.py` derives the data key from it ([lines 714–716 on main as of 2026-10-13](https://github.com/marin-community/marin/blob/5d3df9fb6e6df524ee3dc58db4981ce46b8c5491/experiments/grug/fast_track/train.py#L714-L716)).
  - The "seed noise" here therefore covers init, data order and run-to-run nondeterminism.
  - Two same-seed reruns of the same math in batch 175 differ by about as much as seed pairs do.
- **Dropped pairs.** 6 pairs are left out. Their two seeds differ by 0.03–0.21 on the macro and by about as much on Uncheatable, i.e. broken or diverged runs: `lc1-b77-aol`, `lc1-b77-ple64k`, `lc1-b78-ademamix-cd`, `lc1-b78-ple64-lr1`, `lc1-b87-densegrad` and `lc1-b111-tie-nl`.
  - Rule: drop a pair whose macro seed difference is more than 0.01 from its batch median.
  - Thresholds from 0.005 to 0.02 give the same near-frontier noise shares (35% for `ptb`, 42–43% with `wikitext_103`) and noise cut (17–20%). The SNR gain ranges from 1.35× (0.005, which drops 20 pairs) to 1.48×.
  - Kept in, these six dominate the variance over all pairs, but they hardly change the near-frontier results below, which is why those lead.

## Method
- **Model.** Within a batch, the loss of config A under seed k is μ_A + β_k + ε_{A,k}.
  - A paired delta cancels β_k, so the noise in a two-seed paired Δ has variance var(ε).
  - β is real: seed 1's macro is 0.005 higher than seed 0's in 97% of pairs, mostly from `twitterAAE_HELM_fixed`, `4chan` and `mc4` (Uncheatable: −0.0009). Paired deltas cancel it; unpaired seed comparisons don't.
- **Near the frontier.** These are the configs whose two-seed macro is within 0.01 of their batch median: 305 configs in batches with at least two, 234 degrees of freedom. This is where keep/drop calls happen. It also excludes broken configs without a trimming rule.
- **Signal and noise of a metric.** Centre each config's value within its batch, separately for each seed.
  - The seed-0/seed-1 covariance C01 estimates the variance of the true config effects.
  - (V0 + V1)/2 − C01 estimates the per-seed noise variance, which equals the noise variance of a two-seed paired Δ.
  - Their ratio is the signal-to-noise ratio (SNR). It grows in proportion to the number of seeds averaged.
  - C01 / √(V0·V1) is the seed-to-seed correlation of the config effects.
- **Noise shares.** The same-seed minus the cross-seed covariance gives noise covariances. Subset s's share of the macro's noise variance is noisecov(s, macro) / (16 noisevar(macro)), a covariance (Euler) split that sums to 1.
- **Intervals.** 95% intervals come from 2,000 bootstrap resamples of batches. Ratios are point estimates.

## Findings
1. **`ptb` is 1/16 of the macro but carries about a third of its seed noise** near the frontier: 34.8% (95% CI 28–41%). `ptb` and `wikitext_103` together carry 42.4% (35–50%).
   - Without dropping the six pairs, the figures are 30.9% (23–39%) and 38.1% (30–48%).
   - Averaging the other 14 subsets removes about a third of the noise variance, so the noise SD falls 20%.
   - Over all stable pairs rather than near the frontier, the shares are 32.6% and 39.9%. Over all 404 pairs, they are 8.7% and 17.9%, because the six broken pairs dominate.
2. **Signal and noise near the frontier:**

   | Metric | Seed-to-seed correlation | Signal SD | Noise SD of a two-seed paired Δ | SNR |
   |---|---|---|---|---|
   | Paloma macro, 16 subsets | 0.810 | 0.00325 | 0.00158 | 4.25 |
   | Paloma macro without `ptb`, `wikitext_103` | 0.863 | 0.00315 | 0.00126 | 6.30 |
   | Uncheatable macro | 0.871 | 0.00357 | 0.00137 | 6.77 |
   | `ptb` alone | 0.100 | 0.00436 | 0.01317 | 0.11 |
   | `wikitext_103` alone | 0.481 | 0.00441 | 0.00458 | 0.93 |

3. **Dropping the two cuts the noise SD by 20% and the signal by 3%, so SNR rises 1.48× (95% CI 1.21–1.79).** That is about what 1.5× as many seeds would buy.
   - The noise cut is the most robust number: the noise-SD ratio is 0.78–0.82 for every window, median definition and selection metric tried.
   - The SNR gain moves more:
     - selecting on Uncheatable instead: 1.48× (1.22–1.80);
     - selecting on the 14-subset macro: 1.46× (1.22–1.72);
     - a ±0.02 window: 1.42×;
     - all stable configs: 1.37× (1.18–1.61);
     - near the frontier without dropping the six pairs: 1.38×;
     - near the frontier with the stricter 0.005 trimming threshold: 1.35×.
   - Dropping `ptb` alone gives 1.52×. Dropping `twitterAAE_HELM_fixed` as well would give more, but that subset was chosen after seeing the data, so it isn't proposed.
4. **The 14-subset Δ is the better estimate of the full macro's true change near the bar.**
   - Near the frontier, the two-seed RMSE for the true full-macro effect is 0.00093 with the 14-subset macro, against 0.00112 with the measured full macro.
   - Over all stable configs it is worse (0.00140 vs 0.00122), because for large changes `ptb` and `wikitext_103` move differently from the rest. So the full macro should stay the reference for large changes.
5. **The bar.** Noise alone clears 0.005 on the full macro in 0.70% of within-batch config pairs. A Gaussian with the same SD would give 0.13%, so the tails are heavy.
   - The 14-subset bar with the same empirical rate is 0.0042, both over all configs and near the frontier.
   - The Gaussian scaling 0.005 × 0.80 gives 0.0040.
6. **Two fixed-time calls**, seed 0 / seed 1:

   | Call | Macro, 16 | Without `ptb`, `wikitext_103` | Uncheatable | `ptb` |
   |---|---|---|---|---|
   | Batch 174: candidate 11 vs 10 (2698 vs 2607 steps; same math, 3.5% faster) | −0.0058 / −0.0047 | −0.0075 / −0.0067 | −0.0072 / −0.0070 | +0.0129 / +0.0294 |
   | Batch 200: candidate 12 vs 11 (2526 vs 2698 steps) | −0.0001 / −0.0000 | +0.0017 / +0.0045 | +0.0043 / +0.0063 | −0.0204 / −0.0548 |

   - Batch 200's two sides ran on different clusters, cw-rno2a and cw-us-east-02a.
   - Candidate 11 at equal steps on the two clusters differs by +0.0005 / −0.0008 on the 14-subset macro and −0.0005 / −0.0019 on Uncheatable. `ptb` differs by +0.027 / +0.011 there.
7. **Consistency with the thread's own figures.**
   - The two-seed paired SD of 0.0016 puts the "two-seed noise of ±0.004" (batch 143) at about 2.5 SD.
   - A single-seed paired SD of about 0.0022 fits the "0.003 noise between two runs of the same function" (batch 175).

On the hero, the same two subsets dominate the eval-to-eval noise ([../eval-audit](../eval-audit)).

## Layout
| Path | Contents |
|---|---|
| `code/wb.py` | Minimal read-only client for W&B's public GraphQL endpoint (no key needed) |
| `code/fetch_runs.py` | Fetches every finished `lc1-*` run with its flattened config and final evals |
| `code/pairs.py` | Builds the seed pairs |
| `code/seed_noise.py` | Noise shares over all stable pairs, by period, step count and pairing rule. Writes `results/seed_noise.json` |
| `code/frontier_shares.py` | Noise shares near the frontier for three trimming rules, with bootstrap intervals. Writes `results/frontier_shares.json` |
| `code/snr.py` | Signal, noise and seed-to-seed correlation per metric, for several windows and selection metrics, with bootstrap intervals. Writes `results/snr.json` |
| `code/target_error.py` | Error of each metric as an estimate of the true full-macro effect. Writes `results/target_error.json` |
| `code/tails.py` | Empirical false-win rates and the matching 14-subset bar. Writes `results/tails.json` |
| `code/examples.py` | The batch 174 and 200 calls and the cluster check. Writes `results/examples.json` |
| `code/export_pairs.py` | One row per run of every d512 seed pair, with all 23 subset losses and the dropped flag. Writes `results/pairs_d512.csv` |

## Reproduce
```
cd code
python3 -I fetch_runs.py /tmp/runs.jsonl        # ~130 MB, a few minutes; not redistributed here
for s in seed_noise frontier_shares snr target_error tails examples; do
    python3 -I $s.py /tmp/runs.jsonl /tmp/$s.json
done
python3 -I export_pairs.py /tmp/runs.jsonl /tmp/pairs_d512.csv
```
Runs that finished after the fetch time will change the counts slightly.
