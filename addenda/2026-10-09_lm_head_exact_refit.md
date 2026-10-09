# Addendum, 2026-10-09: the lm_head finding under the Oct 3 refit's exact setting

This addendum does not change the frozen forecast. It makes one finding in [#9977](https://github.com/marin-community/marin/issues/9977) more precise.

**Why.** The issue and Section 4 of `preregistration_v2.md` quote the lm_head effect on the H100 d768–d1536 rungs. On those rungs the fitted floor moves from 1.913 to 1.573 and the forecast from 2.116 to 2.037. The rungs exclude d512 only because the no-lm_head compute table we started from (the pre-registration's) has no d512 entry. As a result, the numbers do not line up with the refit's published 1.858 and 2.100.

**Setting.** This repeats the Oct 3 refit's method exactly, which reproduces its published floor of 1.8581 and finals of 2.1001 (new mix) and 2.1133 (old mix):
- Rungs: H100 ladder, d512–d1536, old and new mixes.
- Progress points: the 15 points from 30% to 100%.
- Fit: one floor shared by all fits, chosen by least squares in loss space.

Only the compute changes. The lm_head FLOPs, 6·d·V·tokens with V = 128,256, are removed from every rung and from the hero.

| Compute | Fitted floor | Floors with SSE within 10% of the minimum | Hero final, new mix | Hero final, old mix |
|---|---|---|---|---|
| With lm_head (the refit) | 1.858 | 1.819–1.895 | 2.100 (2.084–2.116) | 2.113 |
| **Without lm_head** | **1.630** | 1.579–1.678 | **2.056** (2.041–2.070) | 2.072 |

Counting lm_head multiplies each rung's compute by 2.84 at d512, 2.07 at d768, 1.59 at d1024 and 1.33 at d1536, but the hero's only by 1.03.

**Reading.** The refit's final of 2.100 sits 0.061 above the Aug 19 pre-registration (2.039). Removing the lm_head FLOPs alone closes 0.044 of that gap, about 72%. The remaining 0.017 reflects the other differences between the two fits: the ladder, the rungs, and a fitted versus a fixed floor.

**Reproduce.**
```
cd code
python lmhead_exact_refit.py ../data/snapshot_2026-10-09T0440Z ../addenda
```
The output is in `addenda/2026-10-09_lm_head_exact_refit.json`.
