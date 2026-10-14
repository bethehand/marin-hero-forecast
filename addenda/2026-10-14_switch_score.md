# Addendum, 2026-10-14: scored at the 16K switch

The hero switched from 4K to 16K context at step 257,553 ([#10101](https://github.com/marin-community/marin/pull/10101)), and its evals pack at 16K from then on. Condition 2 of `preregistration_v2.md` applies: v2 is scored at the last eval before the switch, and the Aug 19 trajectory is scored at the same eval. Condition 6 excludes every later eval. This addendum does not change the frozen forecast.

**The scored eval.** Step 254,999 (W&B `_step`; #8506 calls it step 255,000) is the parent run's last eval; it stopped at the handoff with no eval after it. Its fraction is p = 255,000 / 390,251 = 0.653426. No PTB exposure window covers it, and no other condition applies: the schedule, optimizer and tokens per update are unchanged.

| Paloma macro (16 subsets), step 254,999 | Median | 80% interval | 95% interval | Quantile score |
|---|---|---|---|---|
| Realized | 2.1114 | | | |
| **v2 (scored)** | 2.1245 | [2.1184, 2.1305] | [2.1153, 2.1336] | **0.0076** |
| v1 | 2.1256 | [2.1197, 2.1314] | [2.1165, 2.1346] | 0.0088 |
| Aug 19 trajectory | 2.1231 | | | 0.0117 |

- The quantile score is 2 × the mean pinball loss over the five frozen quantiles (2.5/10/50/90/97.5%); for a point forecast it equals the absolute error.
- The realized value is 0.0039 below v2's 95% interval, so on coverage this is a miss. v1 misses too.
- v2's quantile score is the lowest of the three thanks to its lower quantiles. Its median (2.1245) missed by 0.0130, more than the Aug 19 point (2.1231) at 0.0117.
- The PTB-free companion, 2.1169, is below v2's companion 95% interval [2.1257, 2.1396] (quantile score 0.0117; v1 0.0130).
- v2 was anchored on the hero's evals through step 227,999; the Aug 19 trajectory predates them.
- Quantiles are interpolated linearly in p between the 0.65 and 0.66 rows of the frozen trajectory files.

**Since the freeze.** No 4K evals follow step 254,999, so this also serves as the 70% check-in (Section 2). Its three items use the frozen v2 code and parameters. The trajectory distribution is redrawn at each new eval's p from the same seed; these are fresh Monte Carlo draws, not the frozen draws themselves. Redrawn this way, the quantiles at p = 0.59–0.66 match the frozen trajectory file within 0.00012, and the PITs carry Monte Carlo error of about ±0.002 near 0.3 and ±0.0003 in the tail.

| Step | Macro (16) | PIT | v2 band | PTB + WikiText | Macro of the other 14 |
|---|---|---|---|---|---|
| 230,999 | 2.1401 | 0.40 | inside 80% | 3.540 | 2.1929 |
| 233,999 | 2.1343 | 0.14 | inside 80% | 3.494 | 2.1896 |
| 236,999 | 2.1249 | 0.004 | below 95% | 3.381 | 2.1870 |
| 239,999 | 2.1204 | 0.0006 | below 95% | 3.318 | 2.1863 |
| 242,999 | 2.1180 | 0.0004 | below 95% | 3.312 | 2.1841 |
| 245,999 | 2.1194 | 0.007 | below 95% | 3.369 | 2.1815 |
| 248,999 | 2.1262 | 0.29 | inside 80% | 3.451 | 2.1834 |
| 251,999 | 2.1234 | 0.25 | inside 80% | 3.469 | 2.1790 |
| 254,999 | 2.1114 | 0.003 | below 95% | 3.300 | 2.1773 |

- All 9 came in below v2's median. 4 fell inside the 80% band and 5 below the 95% band.
- The 5 misses are the 5 evals where PTB and WikiText together were lowest. PTB was low at 236,999 and 242,999, WikiText at 239,999 and 254,999, and both moderately at 245,999.
- The macro of the other 14 subsets fell at 7 of the 9 evals (it rose at 230,999 and 248,999), from 2.1917 at step 227,999 to 2.1773.
- At step 254,999 WikiText fell 0.168 to 1.271, its lowest in the hero's evals (earlier drops: [#9874](https://github.com/marin-community/marin/issues/9874)). That is 0.0105 of the eval's 0.0120 drop.
- Refitted with the 9 new evals (47 evals from p ≥ 0.299 through step 254,999, against 38 at the freeze), the residual slope b goes from +0.0050 to −0.0137. b is the slope of the hero's 15-subset residual against the ladder trajectory, in loss per unit of training fraction, so −0.0137 is about −0.0014 per 10% of training. It fell in all 328 ladder, convention, E and profile rows, by 0.010 to 0.022. Without the two evals with WikiText drops (239,999 and 254,999) it would be −0.0050.

**Reading.** Condition 5 covers PTB only, and no PTB exposure window covers step 254,999, so WikiText's drop stays in the score. In hindsight WikiText deserved a similar rule: in #9874 ClassicLarry suggested scanning it on Oct 7, two days before the freeze, and the eval audit found its drops about eight hours after the freeze. But no WikiText read has been mapped, and adding a rule now would be scoring after the fact. As condition 2 says, the end-of-run value will be reported, unscored.

**Reproduce.**
```
cd code
python score_switch.py ../addenda
python checkin_v2.py ../data/snapshot_2026-10-09T0440Z ../addenda
```
Both read the hero's evals from W&B. The outputs are `addenda/2026-10-14_switch_score.json` and `addenda/2026-10-14_checkin.json`.
