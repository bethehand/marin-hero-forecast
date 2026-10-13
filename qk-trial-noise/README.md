# Reading the qk trial: how noisy are loss and grad norms on shared batches?

This is an independent analysis, not affiliated with Marin or Open Athena. It uses only public W&B runs in `marin-community/marin_moe`.

Context: the hero's context switch will pick `qk_mult` with a short trial ([marin#9615](https://github.com/marin-community/marin/issues/9615)): a one-rack rehearsal (2–3 seeds × 200 steps per arm), then 11-rack arms. Larry noted that the call may rest on "gradient behavior on certain parameters, or stability of loss during its 200 steps" ([comment](https://github.com/marin-community/marin/issues/9615#issuecomment-6150771881)). Runs that already exist show how much each of these moves when nothing but the seed, or nothing at all, changes.

## Data
- **Sept 30 restores**, `mhep-ctx{4k,8k,16k}-s{0,1,2}-20260930`.
  - Each restores the hero's `step-180000` on one GB200 rack and runs steps 180000–180099 at 4,194,304 tokens per step.
  - `qk_mult` is 1.3, 1.39 and 1.48 at 4K, 8K and 16K.
  - `data_seed` is unset, so `trainer.seed` sets the data order.
- **Same-seed reruns**, `ctx9615-{ctl,tip}-{4k,8k,16k}-s0-01` ([#9615 comment](https://github.com/marin-community/marin/issues/9615#issuecomment-6086748179)).
  - Same setup, seed 0, on pre-stack `3a955d678e` (`ctl`) and the optimization stack `a9b3cb8167` (`tip`).
  - With the Sept 30 seed-0 run, each length therefore has three seed-0 runs on three commits. There are no same-commit reruns.
- **67B qk test**, `…cooldown_step39k_yarn_{mscale00,mscale01,mscale02,long_mscale01,long_mscale02}_test20`.
  - 20 steps from step 39000 at 65K packing, seed 0, on TPU.
  - These runs are deterministic: the rerun `long_mscale00` matches `mscale00` exactly.
- Train loss is logged every step. Per-parameter grad norms (`grad/norm/*`, 1,381 keys) are logged every 10 steps. The 49 `router_bias` keys are zero in every run and are left out.

## Findings
1. **Runs with the same seed and packing read the same batches; runs with different seeds don't.**

   | Length | Same seed: per-step SD of the loss difference | Same seed: last-50-step mean difference | Different seeds: per-step SD | Different seeds: last-50-step mean difference |
   |---|---|---|---|---|
   | 4K | 0.00014–0.00027 | ≤ 0.0006 | 0.032 | 0.0007–0.0052 |
   | 8K | 0.00017–0.00047 | ≤ 0.0007 | 0.043 | 0.0002–0.0062 |
   | 16K | 0.00032–0.00045 | ≤ 0.0003 | 0.063 | 0.0056–0.0173 |

   - At step 180000 the loss is identical across the three seed-0 runs at every length, while different seeds differ there by 0.02–0.08. Afterwards the seed-0 runs' per-step losses correlate at 0.9999 or more.
   - At 16K, the largest single-step gap between the seed-0 runs is 0.004, at step 180095. Without that step, the per-step SD is 0.00018–0.00028.
   - The same seed at a different packing does not share batches: across lengths, per-step loss fluctuations of same-seed runs correlate at about 0 (mean 0.035; each r has a standard error of about 0.1). The 4K restores don't share the hero's own batches either (r = 0.09–0.16). So pairing works between arms with the same packing and data order, such as 16K qk arms on the same seed, and it is worth checking.
2. **On shared batches, the per-step loss difference between arms is far steadier than the loss itself.** The 67B arms shared batches: their per-step losses correlate at r = 0.92–1.00. Per-step train-loss difference to the 1.3 arm, while the 1.3 arm's own loss moved by about 0.02 from batch to batch:

   | Arm | First 5 steps, mean | Last 10 steps, mean | Steps below 1.3 |
   |---|---|---|---|
   | all layers 1.570 | −0.0056 | −0.0050 | 19 of 20 |
   | all layers 1.841 | +0.0078 | −0.0026 | 14 of 20 (all from step 39006 on) |
   | long layers ×1.208 | −0.0096 | −0.0038 | 19 of 20 |
   | long layers ×1.416 | −0.0101 | −0.0069 | 20 of 20 |

   - The large swings of the first steps died out within about 5 steps. From step 39005 on, 1.570 stayed 0.003–0.006 below 1.3.
   - From step 39005 on, each arm's difference to 1.3 varied by an SD of 0.0005–0.0019 from step to step, against 0.020 for the 1.3 arm's own loss around its trend. This variation comes from how each arm responds to a given batch; on TPU there is no rerun noise to add.
   - The gaps also drifted by 0.002–0.004 over those 15 steps (long ×1.208 also jumped by 0.006 on one step), as much as the gaps between the middle arms. So an order read from the early steps differs from the late one.
   - Ordered by the last 10 steps: long ×1.416 < 1.570 < long ×1.208 < 1.841 < 1.3. That is the Paloma order at the last eval with the top two swapped; they are 0.0002 apart on Paloma.
   - Uncheatable disagrees: at step 39019, 1.570 and 1.841 are 0.0006 and 0.0012 worse than 1.3 there, and only the long-layer arms are better (−0.0018). Paired train loss is a precise read of the training loss, not a substitute for the evals.
3. **Grad norms reproduce at the first step but not after it.** Gaps are |Δ log grad norm|, shown as percentages; a gap of 1.0 is a factor of 2.7.
   - At step 180000 (same weights, same batch), the three seed-0 16K runs match: the median gap across parameters is 0.13% (90th percentile 1.2%).
   - From step 180010 on, the median gap per parameter is 6–28% at each logged step, and the 90th percentile reaches 1.4 (a factor of 4). Between different seeds the median is 10–20%, so at two of the nine steps the seed-0 runs differ more than different seeds do. Pooled over steps: 11.3% against 14.3%.
   - By parameter type, per layer and pooled over steps, the median gap is 7–20% for the seed-0 runs and 9–26% across seeds. The seed-0 runs are closer for all 24 types, but not by much.
   - Summed over the 48 layers, a few types reproduce better than across seeds: the router 3.4% vs 6.3%, `mlp_gated_norm.w_down` 4.4% vs 11.0%, the shared-expert matrices 3–6% vs 5–7%. Most don't, including `w_q` and `w_k` (16.7% and 20.0% vs 15.0% and 18.3%). The total grad norm differs by 6.1% between the seed-0 runs and 6.8% across seeds.
   - Per parameter, only `output_proj` and the final norms reproduce well: 1–4% between the seed-0 runs against 8–22% across seeds.
   - Caveat: the three seed-0 runs are on three commits. Whether same-commit reruns reproduce better is untested here. GB200 runs are not bitwise deterministic in any case (#9615: the drop fraction varies by up to 6.5% between identical reruns).

## What this suggests for the trial
- Run the 16K arms on the same seeds and read the loss as the per-step difference to the 1.48 arm on the same seed. GB200 reruns of one arm put the nondeterministic part of that difference at 3–5e-4 per step and at most 3e-4 on a 50-step mean. Between different arms, expect it to vary more from step to step, by 0.0005–0.002 on the 67B.
- Check that the arms really share batches: their per-step losses should correlate near 1, as the 67B arms' do (r = 0.92–1.00), rather than near 0 as different seeds' do.
- With unpaired seeds, a 50-step mean at 16K differs by 0.006–0.017 from batch content alone, comparable to or larger than the qk gaps on the 67B (a different model and packing).
- More seeds then show whether an effect holds on other data; they are not needed to estimate the noise.
- The same read should work for the 11-rack arms if they share batches. Nothing here was tested at 11 racks.
- Read the paired loss next to the evals: on the 67B it ordered the arms like Paloma, not like Uncheatable.
- Count a per-layer grad-norm difference between arms only if it is large next to the gap between seed-0 runs (a median of 6–28% per parameter at each logged step), or repeats across seeds and steps. `output_proj` and the final-norm grads are the reproducible ones.

## Layout
| Path | Contents |
|---|---|
| `code/wb.py` | Minimal read-only client for W&B's public GraphQL endpoint |
| `code/pull_loss.py`, `code/pull_reruns.py` | Per-step train loss for the Sept 30 restores, the hero and the `ctx9615` reruns |
| `code/pull_grads.py` | Per-parameter grad norms for the five 16K runs |
| `code/pull_67b.py` | Train loss and evals of the 67B qk arms |
| `code/loss_pairing.py` | Correlation of per-step loss fluctuations within and across seeds and lengths |
| `code/paired_noise.py` | Finding 1 |
| `code/paired_67b.py` | Finding 2 |
| `code/grad_noise_robust.py` | Finding 3 |
| `data/`, `results/` | The pulled series and each script's output |

## Reproduce
```
cd code
python3 pull_loss.py && python3 pull_reruns.py && python3 pull_grads.py && python3 pull_67b.py   # optional: data/ is included
python3 loss_pairing.py && python3 paired_noise.py && python3 paired_67b.py && python3 grad_noise_robust.py
```
