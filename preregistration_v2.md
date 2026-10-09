# Independent forecast of the Marin 535B-A23B hero run's final loss: pre-registration v2

- **Frozen:** 2026-10-09 (v2). This supersedes v1 (tag `prereg-2026-10-09`, frozen earlier the same day), which stays public and unchanged.
- **Data:** the same public-data snapshot as v1, taken at 04:40 UTC, with data through step 227,999 of 390,251 (58.4% of training). At 06:06 UTC the run was at step 229,145 and had logged no newer eval. v2 therefore sees exactly the data v1 saw, and only the method changed.
- **Why a v2:** after publishing v1 we re-audited it line by line (code, numbers and text) and re-ran the backtests. Several modeling choices were not supported by the backtests or the data, and the v1 document had several errors. Section 6 lists every change and Section 7 the errata.
- **Author:** Qirun Li (GitHub @bethehand). The analysis was done with the help of Claude (Anthropic) as a coding and analysis assistant.
- **Hero run:** marin-community/marin [#8435](https://github.com/marin-community/marin/issues/8435).
- **Freeze record:** SHA-256 hashes of every frozen file are in `results/v2/freeze_manifest_v2.json`.

## 1. Forecast

| Target (final eval, 4K eval packing) | Median | 80% interval | 95% interval |
|---|---|---|---|
| **`eval_dropless/paloma/macro_loss`, 16 subsets (primary)** | **2.035** | **[2.023, 2.047]** | **[2.016, 2.054]** |
| Same macro over 15 subsets, excluding PTB (companion) | 2.047 | [2.036, 2.059] | [2.030, 2.065] |

Where the team's point forecasts fall in our predictive distribution for the primary target:

| Team forecast | Value | Percentile in our distribution |
|---|---|---|
| Pre-registered, Aug 19 (#8435) | 2.039 | 68th |
| Swap-adjusted, Sep 14 (W&B) | 2.027 | 20th |
| Refit with fitted floors, new mix, Oct 3 (W&B) | 2.100 | above all 60,000 draws; P(final ≥ 2.07) = 0.0002 |

**In short:** the run is on track to finish at or slightly below the Aug 19 pre-registration. Our median is 0.004 below it. A final loss near 2.10 is very unlikely if the run completes as planned. v1 had a median of 2.045; Section 4 shows where the 0.010 difference comes from.

Trajectory forecasts for every 1% from 59% to 100% are in `results/v2/trajectory_official_macro16_v2.csv` and `trajectory_macro15_v2.csv`. Selected points for the primary target:

| Progress (step) | 70% (273,175) | 80% (312,200) | 90% (351,225) | 100% (390,250) |
|---|---|---|---|---|
| Median | 2.112 | 2.085 | 2.060 | 2.035 |
| 80% interval | [2.105, 2.118] | [2.077, 2.093] | [2.050, 2.070] | [2.023, 2.047] |

## 2. Conditions and scoring rules (fixed in advance)

**Targets**
- **Primary:** `eval_dropless/paloma/macro_loss` at 4K eval packing, read at the final eval of the hero run's W&B lineage (`hero-12d8b6f0-dee637` → … → the current run).
- **Rollbacks:** if part of the run is rolled back and re-run, the value produced by the re-run counts at each step. If W&B keeps the pre-rollback rows instead (as after the Sep 3 rollback within one run, #8506), we use the value the team reports for the re-run; if there is none, we score the next eval after the replayed range.
- **Companion:** the mean of the 15 `eval_dropless/paloma/<subset>-llama3/loss` values, excluding `ptb-llama3`.
- **Steps and fractions:** an eval logged at step s sits at fraction p = (s + 1)/390,251. The final step of the planned schedule is 390,250. Trajectory quantiles are interpolated linearly in p between the rows of the trajectory files. Those rows are every 1%, plus two rows just before and at step 362,280 (see condition 5).

**Conditions**
1. **The run trains at 4K context to the end of the planned 390,251-step schedule:** we score the final eval against the end-of-run distribution.
2. **The training context is extended before the end** (the plan in #9615):
   - Primary score: the last eval before the switch, against our trajectory at that fraction. The team's Aug 19 trajectory (per-5% points, interpolated linearly) is scored at the same eval.
   - The end-of-run value is reported but not scored. If the eval stays pinned at 4K (#9890), we also report the gap between the end-of-run value and our 4K counterfactual.
3. **The token horizon or the learning-rate schedule changes** (shortened, as the policy in #8435 allows, or lengthened):
   - Primary score: the last eval before the schedule changes, against our trajectory.
   - Secondary comparison, for a shortened run that completes its decay: the table in Section 4. For example, ending at 80% gives a median of 2.047.
4. **PTB is dropped from the macro, or the macro definition changes:** the companion target becomes primary.
5. **The eval to be scored falls in the acute window of a PTB exposure:**
   - **The known reads.** #9874 maps the steps at which the hero reads the Paloma PTB text under the current data order: steps 209,390–209,669 (passed) and 362,280–362,578 (ahead, during the cooldown). An exposure's window runs from its start through 12,000 steps after its end. For the second read that is steps 362,280–374,578.
   - **Further copies.** #9874 also notes another PTB copy (n-gram tables, around step 206,300) that its mapping does not cover. It probably caused the dip at eval 206,999, which came before the mapped read. If the team maps that copy, or identifies any other exposure, its window applies too.
   - **Data order changes.** A datamix update under #8435's shortening policy, or a skip to the next shuffle window at a sequence-length change (#9615), can move or remove these reads. Then the windows follow the team's re-mapped steps.
   - Inside a window we score the companion target at that eval and report both.
   - The window is deliberately longer than the effect seen around the first read: three evals (206,999–212,999) sat 0.14–0.34 below the trend, and #9874 reports that the effect had faded by step 216,000.
6. **Any other change to the eval set or eval packing:** only evals before the change are scored.

**Scoring**
- Coverage: does the realized value fall inside the 80% and 95% intervals?
- At the end of the run: the CRPS of our predictive distribution, computed from the frozen Monte Carlo samples in `results/v2/end_samples_v2.npz`, against the absolute error of each team point forecast. For a point forecast CRPS equals the absolute error, so the comparison is like for like.
- At a trajectory point (conditions 2 and 3): the quantile score, 2 × the mean pinball loss over our five frozen quantiles (2.5/10/50/90/97.5%). For a point forecast it also equals the absolute error.
- v2 is the scored forecast. We will report v1's score alongside it.
- The outcome will be published whether the forecast turns out good or bad.

**Mid-course update**
At 70% (step ~273,175) we will re-run the frozen v2 code on new data and post an addendum. It will give:
- the PIT of each new eval under the trajectory distribution;
- how many new evals fall in the 80% band;
- the re-estimated residual slope b.

The scored forecast remains the frozen one.

## 3. Method

**Anchoring.** We use the scaling ladder only for *how much more the loss will fall from now to the end*. The level comes from the hero's own evals:

```
final = f(1) + a + κ·b·(1 − p_now) + ε − β + η
```

- **f(p):** the hero trajectory predicted from the ladder. It uses the same per-5% log-space fits as the team's pre-registration (`L = E + A·C^(−α)` with E fixed), smoothed with a cubic in p over 30%–100%.
- **a, b:** offset and slope of the hero's residuals against f. They are fitted on the 38 evals with p ≥ 0.299 (steps 116,999–227,999). The data swap was at step 108,000; the two post-swap evals before the cutoff (steps 110,999 and 113,999) are left out.
- **κ ~ U[0, 0.2]:** how much of the observed residual drift persists (v1: U[0, 1]). It is set by backtest: 0.2 is the largest κ whose RMS error over the 15-subset pseudo-heroes is within 10% of the minimum.

  | κ | 0 | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 | 1 |
  |---|---|---|---|---|---|---|---|
  | RMS error | 0.0066 | 0.0050 | 0.0049 | 0.0053 | 0.0072 | 0.0127 | 0.0284 |

- **ε:** method error, Student-t with 8 degrees of freedom and scale 0.0054 (SD 0.0062). The scale is the RMS backtest error (Section 4), taken around zero rather than around the distance trend below, which errs on the wide side.
- **β:** distance-bias allowance, β ~ U[0, b(d)]. Here b(d) is the backtest error trend evaluated at each ladder's extrapolation distance: 0.0072 at 337× (GB200) and 0.0106 at 1,890× (H100). See "Extrapolation distance" below.
- **η:** eval noise at the final eval, SD 0.0031 for the 15-subset target.
- **On the trajectory:** ε and β describe errors in the remaining drop, so they enter in proportion to (p − p_now)/(1 − p_now). The shortened-horizon table applies them in full, because there the run's own end is the target.

**Ladders and the d2048 extension.**
- GB200 d768–d2048 has weight 0.75 and the same batch-scaling recipe as the hero.
- H100 d512–d1536 on the new mix has weight 0.25 and a similar data schedule: its mix switches at 25% of training, against the hero's 27.7% (#9126).
- The d2048 rung stopped at 81%. Beyond 80% we extend it along its own 60–80% linear trend, plus the deviation that each completed GB200 rung (d768, d1024, d1536) showed from *its* 60–80% trend.
- That gives three profiles with equal weight. For the 15-subset macro they are about 0 at 85–95% and +0.001 to +0.003 at 100%. v1 instead added U[0.003, 0.006] at every point beyond 80%.

**Uncertainty we sample over (60,000 draws):**
- ladder;
- compute convention: with or without lm_head FLOPs, 0.5 each;
- E ~ U[1.3, 1.7] without lm_head, U[1.5, 1.9] with it;
- d2048 profile;
- estimation error of a and b;
- κ, ε, β and η.

**Extrapolation distance.** The hero's compute is 337× the largest GB200 rung and 1,890× the largest H100 rung, while the backtests reach 6.5–166×.
- **Backtest errors drift pessimistic with distance, but the evidence is weak.**
  - The mean error is −0.0005 at 6.5–8×, +0.0031 at 52× and +0.0069 for the 67B at 166×.
  - A linear fit in log10 distance gives +0.0046 per decade (SE 0.0024, clustered by case with a small-sample correction). Fitting the 9 case means instead gives the same slope with SE 0.0034, t = 1.35. Without the 67B the slope is +0.0034 (case-mean t = 0.66).
  - The two 52× cases share their training rungs (H100 d512 + d768), and their targets also appear among the 6.5–8× cases.
  - The 67B differs from the ladder cases in method, target, eval length (8K) and endpoint (a mid-schedule point, not the end of a decay).
- **The drift tracks the floor E.** It grows with E and is larger when lm_head FLOPs are counted. For the 15-subset macro at 52×, the mean error is +0.0003 at E = 1.3 without lm_head and +0.0073 at E = 1.9 with it. This fits the 67B backtest, where E at the low end of our priors fit best. β is nevertheless applied the same way to every E and convention draw.
- **Checks within the hero agree only partly.** Without β, the two full ladders give the same primary median (2.0386), even though they extrapolate 337× and 1,890×. Dropping rungs to extrapolate further gives the 15-subset centers below. The H100 ladder rises by about +0.003 per decade, while the GB200 ladder does not.

  | Ladder (rungs) | Extrapolation | 15-subset center |
  |---|---|---|
  | GB200 (d768–d2048) | 337× | 2.0512 |
  | GB200 (d768–d1536) | 1,890× | 2.0511 |
  | GB200 (d768, d1024) | 15,126× | 2.0433 |
  | H100 (d512–d1536) | 1,890× | 2.0514 |
  | H100 (d512–d1024) | 15,126× | 2.0542 |
  | H100 (d512, d768) | 98,536× | 2.0566 |

Because the trend is suggestive but not significant, we hedge rather than apply it in full: β ~ U[0, b(d)]. One end assumes no distance bias; the other applies the full backtest trend, which b(d) extrapolates 0.3–1.1 decades beyond the 67B. The allowance lowers the median by 0.004. Applying none of it gives 2.039; applying all of it gives 2.031.

**PTB.** The 16-subset macro is forecast as `macro15 + D/16`, where D = PTB − macro15 (an exact identity, checked on the ladder rungs).
- **Current level:** a linear fit to the 35 clean evals with p ≥ 0.299 puts D at −0.123 now, with scatter SD 0.042.
  - The fit leaves out the three evals around the first read (206,999–212,999).
  - Its residuals have lag-1 autocorrelation 0.50. We therefore widen the level's standard error from 0.015 (ordinary least squares) to 0.025, a factor of √((1 + ρ)/(1 − ρ)).
- **Remaining change:** taken from 10 ladder rungs (GB200, H100 new mix and H100 old mix): mean −0.048, SD 0.016, widened to 0.02.
- **Did the first exposure leave a lasting shift? Unclear.**
  - The 5 evals since average −0.149. The line fitted to the 30 clean evals before the exposure is flat at about −0.088.
  - But that earlier series is noisy (residual SD 0.042, lag-1 autocorrelation 0.47). The four evals at steps 191,999–200,999 also averaged −0.150.
  - The 5 recent evals sit 0.029 below the linear fit, about one standard error once autocorrelation is allowed for.
  - So we add no allowance for the first exposure beyond the wider standard error.
- **Second exposure (steps 362,280–362,578):** an allowance r2 ~ U[−0.06, 0] covers a possible lasting effect, up to the size of the gap between the recent evals and the pre-exposure line. It applies from step 362,280 on: in the end-of-run samples, at the trajectory rows from step 362,280, and in shortened runs that reach that step.
- Anchoring the 16-subset macro directly, rather than through D, gives a median of 2.034. That variant is reported for reference.

**Perturbations checked in v1, unchanged** (numbers in `results/step5/perturbations_step4.json`):
- **The Sep 15 data swap:** absorbed by anchoring after the swap.
- **The cooldown mix:** the remaining drops of the two ladders agree at matched widths.
- **Extra token drops at 16K:** about +0.0005.
- **qk_mult and eval-packing effects of a context switch:** handled by the conditions above.

## 4. Evidence

**Reproduction.** Both team forecasts reproduce from public data:
- The pre-registration gives 2.0388 against 2.039. All 20 per-5% points agree to within 0.00054, about the rounding of the three decimals logged on W&B.
- The Oct 3 refit gives a floor of 1.8581 and ends of 2.1001 / 2.1133, matching exactly.

**Why 2.04 and 2.10 differ.** The refit's compute includes lm_head FLOPs.
- On the same H100 runs (d768–d1536), excluding lm_head FLOPs moves the fitted floor from 1.913 to 1.573. It moves the end-of-run forecast from 2.116 to 2.037.
- The lm_head share falls with width: the with/without ratio is 2.07 at d768, 1.33 at d1536 and 1.03 at the hero. That changes the curvature the floor is fitted from.

**67B-A2B backtest (#6044).** With 3 anchors, the known values at 21%, 63% and 80% are reached by extrapolating about 166×:
- Fixed E lands within about 0.008: E = 1.3 without lm_head, or E = 1.5 with it.
- A shared free floor is pessimistic by 0.05–0.10.

**Method error.**
- Errors are prediction minus actual, with κ = 0.1, averaged over E and the two compute conventions.
- The 8 ladder cases are each anchored at 55% and at 60% of their run and predict the run's final eval.
- The 67B is anchored at 21% and at 63% and predicts its 80% value. Only these three points are published, so it uses a simpler form of the method: a single-point anchor, no smoothing, no drift term, and the 16-subset macro.

| Backtest cases | Extrapolation | Errors | RMS | Mean |
|---|---|---|---|---|
| 6 same-ladder and cross-ladder cases | 6.5–8× | 12 | 0.0053 | −0.0005 |
| 2 long-range cases (H100 d1536, and GB200 d1536, each from H100 d512 + d768) | 52× | 4 | 0.0038 | +0.0031 |
| 67B-A2B | 166× | 2 | 0.0079 | +0.0069 |
| **Pool used for ε and the distance trend** | | 18 | **0.0054** | +0.0011 |

- The 8 ladder cases predict only 4 distinct target rungs. The largest is d1536, with about 481M active parameters.
- On the 8 ladder cases, the RMS error is 0.0050 for the 15-subset macro and 0.0051 for the primary composite. The composite here is the 15-subset forecast + D/16, with D's remaining change taken from the other rungs. The direct 16-subset variant has an RMS error of 0.0068.
- The composite is as accurate as its 15-subset part. The ladder rungs had no PTB exposure, so this does not test the r2 allowance.

**Calibration (nested leave-one-case-out).** Each case is held out with both its anchor points.
- κ_max, the ε scale and the distance trend are re-estimated from the remaining cases. v1 already re-estimated ε this way; doing the same for κ_max and the trend is new.
- The held-out case then gets the hero's Monte Carlo, restricted to its own training ladder: an E grid with step 0.05, no d2048 profile, and no PTB allowance.

| Target | Inside 80% interval | Inside 95% interval | Mean PIT |
|---|---|---|---|
| 15-subset macro | 14 of 16 | 16 of 16 | 0.50 |
| Primary composite | 15 of 16 | 16 of 16 | 0.45 |
| Direct 16-subset variant (incl. the 67B) | 16 of 18 | 18 of 18 | 0.45 |

- These are 8–9 cases with two anchor points each, and several cases share a target rung, so the folds are not independent and the counts are coarse.
- The 15-subset misses are both the H100 old-mix rung. That ladder is not used for f(p), but it does feed κ_max, ε, the distance trend and 3 of the 10 ΔD rungs.

**From v1 to v2, one change at a time.** Same data; one seed and 60,000 draws for every row. That seed differs from the frozen forecast's, so the last row differs from it by Monte Carlo noise (at most 0.0002).

| Step | Primary median | 80% interval | 15-subset median |
|---|---|---|---|
| v1 method, re-run | 2.0450 | [2.0238, 2.0581] | 2.0577 |
| + κ ~ U[0, 0.2] | 2.0418 | [2.0305, 2.0534] | 2.0544 |
| + empirical d2048 profiles | 2.0386 | [2.0276, 2.0501] | 2.0510 |
| + method error pooled with the 67B, Student-t | 2.0385 | [2.0273, 2.0504] | 2.0511 |
| + distance-bias allowance | 2.0346 | [2.0228, 2.0468] | 2.0471 |
| + PTB: allowance at the second exposure, wider level SE (v2) | 2.0346 | [2.0226, 2.0469] | 2.0471 |

**Sensitivity of the primary target.**
- Each row changes one setting relative to the baseline. All rows use the same seed and 40,000 draws. Rows that change how many random numbers a draw consumes (κ fixed, Normal ε, ν = 7) shift the later draws, which adds about 0.0001 of Monte Carlo noise.
- Dropping a ladder's largest rung also raises that ladder's β cap, because the extrapolation distance grows. Without β, dropping the rung moves the median by −0.0002 for GB200 (2.0386 → 2.0384) and by +0.0028 for H100 (2.0386 → 2.0414).

| Variant | Median | 80% interval |
|---|---|---|
| v2 baseline | 2.0346 | [2.0226, 2.0470] |
| κ = 0 (no drift) | 2.0342 | [2.0222, 2.0470] |
| κ ~ U[0, 1] (v1 prior) | 2.0372 | [2.0174, 2.0505] |
| GB200 ladder only (337×) | 2.0349 | [2.0233, 2.0469] |
| H100 new-mix ladder only (1,890×) | 2.0333 | [2.0203, 2.0470] |
| GB200 without d2048 (1,890×) | 2.0331 | [2.0206, 2.0459] |
| H100 without d1536 (15,126×) | 2.0341 | [2.0196, 2.0492] |
| Ladder weights 50/50 | 2.0343 | [2.0218, 2.0470] |
| Normal instead of Student-t | 2.0347 | [2.0232, 2.0465] |
| Method error and distance trend without the 67B | 2.0356 | [2.0240, 2.0475] |
| No distance-bias allowance | 2.0386 | [2.0271, 2.0505] |
| Full distance trend (β = b(d)) | 2.0307 | [2.0189, 2.0426] |
| PTB allowance as in v1 (U[−0.06, 0], phased in) | 2.0346 | [2.0227, 2.0468] |
| No PTB allowance | 2.0365 | [2.0246, 2.0487] |
| Quadratic smoothing | 2.0347 | [2.0227, 2.0471] |
| Quartic smoothing | 2.0336 | [2.0215, 2.0460] |
| No smoothing (linear interpolation of the per-5% fits) | 2.0346 | [2.0225, 2.0469] |
| Anchor window from 45% | 2.0317 | [2.0198, 2.0441] |

Across these variants the median stays between 2.031 and 2.039.

**Shortened horizon with a completed decay** (secondary comparison for condition 3; assumes the current data order):

| Run ends at | 80% | 85% | 90% | 95% | 100% |
|---|---|---|---|---|---|
| Last step | 312,200 | 331,712 | 351,225 | 370,737 | 390,250 |
| Primary median | 2.047 | 2.044 | 2.041 | 2.037* | 2.035 |
| 80% interval | [2.035, 2.058] | [2.032, 2.055] | [2.030, 2.053] | [2.025, 2.049]* | [2.023, 2.047] |

\* A run ending at 95% would end inside the second exposure's acute window. Condition 5 would then score the companion target, whose median there is 2.050.

## 5. Limitations
- **Extrapolation distance.** The hero extrapolates 337–1,890× beyond the largest rungs, while the backtests reach 166×. The distance trend is not significant at the case level and is extrapolated beyond its data; the uniform hedge is a judgment.
- **Few cases.** There are 9 backtest cases, and several share a target rung. The size of ε is therefore itself uncertain, which is why ε has Student-t tails.
- **Judgment calls:** the 10% tolerance that sets κ_max, the uniform form of the distance allowance, and the PTB allowance, which rests on a single earlier exposure. The sensitivity table shows each.
- **Uniform β:** β does not vary with E or the compute convention, although the backtests suggest the bias is concentrated in high-E and lm_head draws.
- **Anchor regression:** the regression for a and b uses ordinary least squares and ignores autocorrelation. Only the PTB level's standard error is widened for it.
- **Context switch:** its effects on 4K-packed evals are outside the model and are handled by the conditions.
- **Revision timing:** v2 was written after v1 was published, but before any newer hero eval existed.
  - Changes 1–4 come from backtests on the ladders and the 67B.
  - Change 5 comes from the hero's own PTB evals, all of which were already in v1's snapshot, and from the exposure steps in #9874.
  - #9874's step mapping is the team's; we have not checked it independently.

## 6. Changes from v1
1. **Drift prior κ:** U[0, 1] → U[0, 0.2].
   - Why: in backtests of the same estimator, the error is smallest near κ = 0.15 and grows quickly beyond that (RMS 0.0127 at κ = 0.5, 0.0284 at κ = 1). v1's prior was a judgment.
   - Effect: median −0.003; most of the narrower interval.
2. **d2048 beyond 80%:** constant +U[0.003, 0.006] → empirical profiles from the completed GB200 rungs.
   - Why: those rungs show essentially no deviation at 85–95%, so v1 over-corrected.
   - Effect: −0.003.
3. **Method error:**
   - The backtest now scores the same estimator as the forecast; v1 scored it with κ = 0 but forecast with κ ~ U[0, 1].
   - Added two 52× pseudo-heroes, and the 67B now enters the error pool for the 15-subset target.
   - ε is Student-t with ν = 8 instead of Normal.
   - Effect: ε is slightly wider (SD 0.0057 → 0.0062) with heavier tails; the median does not change.
4. **Distance-bias allowance (new):** β ~ U[0, b(d)], from the backtest error trend in log distance.
   - Why: v1 treated the method error as independent of distance, but the backtests drift pessimistic as distance grows, though not significantly.
   - Effect: −0.004.
5. **PTB:**
   - The allowance U[−0.06, 0], which v1 phased in linearly, is now r2 ~ U[−0.06, 0], applied from the second exposure at step 362,280.
   - The current level's standard error is widened for autocorrelation (0.015 → 0.025).
   - Why: #9874 dates the second exposure, and the hero's PTB gap is autocorrelated.
   - Effect: none on the end-of-run median. Trajectory points before step 362,280, and shortened runs that end before it, no longer carry the allowance.
6. **Validation:**
   - The primary composite target is now backtested and calibrated itself; in v1 only the 15-subset and the direct 16-subset variants were.
   - Each leave-one-out fold now also re-selects κ_max and the distance trend.
   - The 67B branch now includes eval noise.
7. **Shortened-horizon table:** a run that completes its decay early now gets the full remaining change in D. v1 scaled it by the remaining progress, which made its 80% entry about 0.001 too high.
8. **Code, no numerical effect on this data:**
   - Duplicate steps across runs of the lineage now take the most recent run (v1 took the earliest).
   - Trajectory files now use the step convention in Section 2, with fractions to six decimals.
9. **Scoring rules, made more precise:**
   - step-to-fraction convention and interpolation;
   - the rollback rule;
   - lengthened horizons and other schedule changes;
   - the PTB-exposure rule;
   - the quantile score at trajectory points.

## 7. Errata for the v1 document
The v1 forecast table matched its frozen results. These statements in the text did not:
- "an 80% interval of ±0.017": the interval was asymmetric, −0.021 / +0.013 around the median.
- The trajectory table at 80% progress: the lower 80% bound was 2.079, not 2.080.
- "fitted on all evals after the data swap (steps 116,999–227,999, 38 evals)": the fit uses p ≥ 0.299; the two post-swap evals before that are excluded.
- "In our analysis at 57.7% of training … span about 0.2 … 0.086–0.109": these numbers came from an exploratory run on an earlier snapshot and are not in the published results.
- "PTB alone has eval-to-eval SD ≈ 0.066": that figure included the three contaminated evals; without them it is about 0.034.
- "that variant gives a median of 2.042": it was 2.04148, which rounds to 2.041.
- "all 20 per-5% points agree to within 0.0005": the largest difference was 0.00054.
- "(… plus the 67B) has an RMS of 0.006 for the 15-subset macro": the 67B entered only the 16-subset figure.
- "80% intervals cover … 11 of 14 (16-subset)": this was the direct 16-subset variant, not the primary composite. The same applies to the "79–83%" in the v1 README.
- "ending at 80% with a completed decay gives a median of 2.055": about 0.001 too high from the ΔD scaling (see change 7).
- "The largest pseudo-hero has 1.2B active parameters": the largest pseudo-hero target is d1536, about 481M active. The 1.2B rung (d2048) was only used as a training rung.
- `results/hero_actual.csv`, `results/step2_*` and `results/step3/` were produced from an earlier snapshot (data through step 224,999), not the frozen one.
- `code/check_snapshot.py` needs the full snapshot, which is not redistributed, so it fails on the published data.

## 8. Reproduce
```
pip install -r requirements.txt
cd code
python forecast_v2.py ../data/snapshot_2026-10-09T0440Z /tmp/v2
```
- The files in `/tmp/v2` match `results/v2/` byte for byte on our platform: macOS arm64, Python 3.11.15, numpy 2.4.6. On other platforms the last digits of floating-point results can differ.
- The hashes are in `results/v2/freeze_manifest_v2.json`.
- The v1 files (`05_preregistration.md`, `results/step5/`, `code/freeze_step5.py`) are kept unchanged. The v1 freeze can still be checked against `results/step5/freeze_manifest.json`.
