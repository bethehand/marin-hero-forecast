# Eval audit: PTB and WikiText-103 drops on the hero, and a decon recall test

This is an independent analysis, not affiliated with Marin or Open Athena. It uses the same public W&B snapshot as the forecast (`../data/snapshot_2026-10-09T0440Z`, evals through step 227,999) and public GitHub files. It does not change the frozen forecast.

## Findings
1. **Noise attribution.** After the 108k mix swap, PTB accounts for 50–68% of the variance in the Paloma macro's eval-to-eval movement, and WikiText-103 for 19–27%. The range covers three detrending methods; the block-bootstrap 95% CI for PTB is 0.33–0.85. Dropping both subsets cuts the macro's noise SD to 43–61% of its current value. The 7 Uncheatable subsets show no drop events at z = 4, 5 or 6.
2. **PTB drops not explained by the reads mapped in marin#9874.** At 104,999, PTB sits 0.165 below the average of its two neighbours (2.143 → 1.961 → 2.109); there is no mapped read nearby. At 191,999 it drops 0.10 and stays below 2.08 through 227,999. The 176,080–176,355 read, which has only PTB test text, moves the eval by 0.029.
3. **WikiText-103 drops that general Wikipedia doesn't show.** At 128,999, wikitext_103 falls 0.149 from the previous eval while `m2d2_wikipedia_unsplit` falls 0.0015. Smaller one-sided drops recur at 137,999, 152,999, 203,999 and 212,999–215,999.
4. **Decon recall on the torch-nplm document.** We indexed Paloma PTB, which the real eval corpus doesn't include. Marin's matcher, run unchanged with the production settings, scores the trained document (PTB validation set as 6-word sliding windows) at overlap 0.000. The same repo with the raw `valid.txt` scores 1.000. Windows of any width from 3 to 30 stay at or below 0.144. A de-windowing pre-pass, which stitches shift-by-one lines back into text, recovers 1.000. In 5,724 ordinary files it triggered on 0 of 241,449 paragraphs.

![PTB and WikiText-103](results/post/fig_9874.png)

## Layout
| Path | Contents |
|---|---|
| `code/analyze_eval_events.py` | Noise attribution, the pre-specified event detector, and the post-hoc two-sided spike statistic. Writes `results/phase1/` |
| `code/post_9874_figure.py` | The quoted eval values (`numbers_9874.json`) and the figure. Writes `results/post/` |
| `code/decon_recall_windowed.py` | Loads the matching functions of Marin's `decon.py` from a checkout, unmodified, and runs the recall test and the de-windowing prototype. Writes `results/decon/` |
| `code/fetch_inputs.sh` | Fetches the third-party inputs at pinned commits: torch-nplm, an independent PTB copy, and a Marin checkout. They are not redistributed here |

## Reproduce
```
pip install -r ../requirements.txt
cd code
python analyze_eval_events.py ../../data/snapshot_2026-10-09T0440Z /tmp/phase1
python post_9874_figure.py ../../data/snapshot_2026-10-09T0440Z /tmp/post
sh fetch_inputs.sh ../inputs
python3.12 -I decon_recall_windowed.py ../inputs/marin ../inputs/torch-nplm ../inputs/wojzaremba-lstm/ptb.valid.txt /tmp/decon \
    ../inputs/marin "$(python3.12 -c 'import sysconfig; print(sysconfig.get_paths()["stdlib"])')"
```
- The decon test needs Python 3.10 or later and nothing outside the standard library.
- The trailing arguments are the directories used as the negative control for the de-windowing pass. The timings in `decon_recall_windowed.json` depend on the machine.
- `decon.py` at Marin commit `3c62e65f8f` has SHA-256 `a6ef1ed28f95d72259fa5a16540b4fdf5f82546785c20da16ea8963130f626cb`. The torch-nplm `valid.txt` is byte-identical to `ptb.valid.txt` in wojzaremba/lstm.
