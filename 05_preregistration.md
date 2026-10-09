# Independent forecast of the Marin 535B-A23B hero run's final loss (pre-registration)

- **Frozen:** 2026-10-09, from a public-data snapshot taken at 04:40 UTC.
- **Data through:** step 227,999 of 390,251 (58.4% of training).
- **Author:** Qirun Li (GitHub @bethehand). The analysis was done with the help of Claude (Anthropic) as a coding and analysis assistant.
- **Hero run:** marin-community/marin [#8435](https://github.com/marin-community/marin/issues/8435).
- **Freeze record:** SHA-256 hashes of every frozen file are in `results/step5/freeze_manifest.json`.

## 1. Forecast

| Target (final eval, 4K eval packing) | Median | 80% interval | 95% interval |
|---|---|---|---|
| **`eval_dropless/paloma/macro_loss`, 16 subsets (primary)** | **2.045** | **[2.024, 2.058]** | **[2.012, 2.064]** |
| Same macro over 15 subsets, excluding PTB (companion) | 2.058 | [2.037, 2.070] | [2.025, 2.076] |

How the team's point forecasts sit in our predictive distribution for the primary target:

| Team forecast | Value | Percentile in our distribution |
|---|---|---|
| Pre-registered, Aug 19 (#8435) | 2.039 | 32nd |
| Swap-adjusted, Sep 14 (W&B) | 2.027 | 13th |
| Refit with fitted floors, new mix, Oct 3 (W&B) | 2.100 | above the 99.99th (no Monte Carlo draw reached it) |

**In short: the run is most likely to finish close to the Aug 19 pre-registration, about 0.006 worse, with an 80% interval of ±0.017. A final loss near 2.10 is very unlikely if the run completes as planned.**

Trajectory forecasts for every 1% from 59% to 100% are in `results/step5/trajectory_official_macro16.csv` and `trajectory_macro15.csv`. Selected points for the primary target:

| Progress (step) | 70% (273,176) | 80% (312,201) | 90% (351,226) | 100% (390,251) |
|---|---|---|---|---|
| Median | 2.114 | 2.089 | 2.066 | 2.045 |
| 80% interval | [2.107, 2.120] | [2.080, 2.097] | [2.052, 2.076] | [2.024, 2.058] |

## 2. Conditions and scoring rules (fixed in advance)

**Targets**
- **Primary:** `eval_dropless/paloma/macro_loss` at 4K eval packing, read at the final eval of the hero run's W&B lineage (`hero-12d8b6f0-dee637` → … → the current run).
- **Companion:** the mean of the 15 `eval_dropless/paloma/<subset>-llama3/loss` values, excluding `ptb-llama3`.

**Conditions**
1. **The run trains at 4K context through step 390,251:** we score the final eval.
2. **The training context is extended before the end** (the plan in #9615):
   - Primary score: the last eval before the switch, against our trajectory linearly interpolated at that step. The team's pre-registered trajectory is compared at the same point.
   - The end-of-run value is reported but not scored. If the eval stays pinned at 4K (#9890), we also report the gap between the end-of-run value and our 4K counterfactual.
3. **The token horizon is shortened** (the policy in #8435):
   - Primary score: the last eval before the schedule changes.
   - Secondary comparison: the approximate shortened-horizon table in `results/step5/perturbations_step4.json`. For example, ending at 80% with a completed decay gives a median of 2.055.
4. **PTB is dropped from the macro, or the macro definition changes:** the companion target becomes primary.
5. **Any other change to the eval set or eval packing:** only evals before the change are scored.

**Scoring**
- Coverage: does the realized value fall inside the 80% and 95% intervals?
- CRPS of our predictive distribution, computed from the frozen Monte Carlo samples in `results/step5/end_samples.npz`, compared with the absolute error of each team point forecast. CRPS reduces to absolute error for a point forecast, so the comparison is like for like.
- The outcome will be published whether the forecast turns out good or bad.

**Mid-course update**
At 70% (step ~273,176) we will re-run the same frozen code on new data and post an addendum. It will report whether the observed evals stayed inside the trajectory intervals and whether the residual drift (Section 3) continued. The scored forecast remains the frozen one.

## 3. Method

**Anchoring.** We use the scaling ladder only for *how much more the loss will fall from now to the end*. The level comes from the hero's own evals:

```
final = f(1) + a + κ·b·(1 − p_now) + ε + η
```

- **f(p):** the hero trajectory predicted from the ladder, using the same per-5% log-space fits as the pre-registration (`L = E + A·C^(−α)` with E fixed), smoothed with a cubic in p over 30%–100%.
- **a, b:** offset and slope of the hero's residuals against f, fitted on all evals after the data swap (steps 116,999–227,999, 38 evals).
- **κ ~ U[0, 1]:** how much of the observed residual drift persists.
- **ε:** method error, set from backtests.
- **η:** eval noise at the final eval.

**Why anchoring.** In our analysis at 57.7% of training, across fixed E between 1.2 and 1.9 and both compute conventions, absolute extrapolations of the final value span about 0.2. The predicted *remaining drop* spans only 0.086–0.109. Anchoring therefore removes most of the dependence on E, which a ladder of four small models cannot pin down.

**Uncertainty we sample over (60,000 draws):**
- Ladder: GB200 d768–d2048, weight 0.75, the same batch-scaling recipe as the hero; H100 d512–d1536 new mix, weight 0.25, the same data schedule as the hero.
- Compute convention: with or without lm_head FLOPs, weight 0.5 each.
- E ~ U[1.3, 1.7] without lm_head, U[1.5, 1.9] with lm_head.
- Correction to the d2048 extrapolation beyond 80% ~ U[0.003, 0.006].
- Estimation error of the anchor offset and slope.
- Drift persistence κ.
- Method error ε: SD 0.0057 for the 15-subset target.
- Final-eval noise: SD 0.0031 for the 15-subset target.

**PTB.** The 16-subset macro is forecast as `macro15 + (PTB − macro15)/16`:
- The anchored PTB gap leaves out the three contaminated evals (steps 206,999, 209,999 and 212,999; see #9874).
- Its remaining change comes from 10 ladder rungs: mean −0.048, SD 0.016, widened to 0.02.
- An allowance of U[−0.06, 0] covers any lasting effect of the two exposures.
- PTB alone has eval-to-eval SD ≈ 0.066. Anchoring the 16-subset macro directly is sensitive to the contamination dip; that variant gives a median of 2.042 and is reported for reference.

**Perturbations checked** (numbers in `results/step5/perturbations_step4.json` and `remaining_drop_by_ladder.csv`):
- **The Sep 15 data swap:** absorbed, because anchoring uses only post-swap evals.
- **The new cooldown mix:** remaining drops at matched widths agree between the GB200 ladder and the H100 new-mix ladder (d768: 0.186 vs 0.187), so no adjustment is needed.
- **Extra token drops at 16K:** about +0.0005 nat, based on the loss change at the step-81,716 transport switch.
- **The effects of qk_mult and eval packing at a context switch:** not modeled; handled by the conditions above.

## 4. Evidence

**Reproduction.** Both team forecasts reproduce from public data:
- The pre-registration gives 2.0388 against 2.039; all 20 per-5% points agree to within 0.0005.
- The Oct 3 refit gives a floor of 1.8581 and ends of 2.1001 / 2.1133, matching exactly.

**Why 2.04 and 2.10 differ.** The refit's compute includes lm_head FLOPs. On the same H100 runs (d768–d1536), excluding lm_head FLOPs moves the fitted floor from 1.913 to 1.573 and the end-of-run forecast from 2.116 to 2.037. The lm_head share falls with width: with/without ratio 2.07 at d768, 1.33 at d1536 and 1.03 at the hero. That changes the curvature the floor is fitted from.

**67B-A2B backtest (#6044).** With 3 anchors, the known targets at 21%, 63% and 80% are reached by extrapolating about 166×:
- Fixed E lands within ~0.008: E = 1.3 without lm_head, or E = 1.5 with it.
- A shared free floor is pessimistic by 0.05–0.10.

**Method error.** The anchored method's mean error on held-out pseudo-heroes (same-ladder and cross-ladder rungs, plus the 67B) has an RMS of 0.006 for the 15-subset macro and 0.007 for the 16-subset macro.

**Calibration (leave one case out).**
- 80% intervals cover 10 of 12 cases (15-subset) and 11 of 14 (16-subset).
- 95% intervals cover all cases.

**Robustness.** The 15-subset median stays between 2.040 and 2.064 across nine variants:
- smoothing degree;
- no smoothing;
- anchor window starting at 45%;
- GB200 only, H100 only, or 50/50;
- κ fixed at 0 or at 1.

That corresponds to roughly 2.03–2.05 for the primary target.

## 5. Limitations
- The largest pseudo-hero has 1.2B active parameters; the only large-scale check is the 67B-A2B run.
- The drift-persistence prior κ ~ U[0, 1] is a judgment, not an estimate.
- The anchor regression uses ordinary least squares and ignores autocorrelation, though this term is small.
- The PTB treatment relies on rung-level gap dynamics and an assumed allowance for lasting contamination effects.
- Effects of a context switch on 4K-packed evals (qk_mult, distribution shift) are outside the model and handled by the conditions.

## 6. Reproduce
```
python code/fetch_snapshot.py --out data/<new-snapshot>   # public W&B / GitHub / GCS data, read-only
python code/freeze_step5.py data/snapshot_2026-10-09T0440Z results/step5
```
- Code: `code/`.
- Frozen data snapshot: `data/snapshot_2026-10-09T0440Z/`; its `manifest.json` holds per-file SHA-256 hashes.
- Results: `results/step5/`.
- Earlier steps: reproduction in `results/step2_*`, exploration in `results/step3_explore/`, and the step 3 run in `results/step3/`.
