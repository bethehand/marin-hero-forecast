"""Empirical false-win rates: how often noise alone moves a two-seed paired Δ past a bar.

For two configs A, B in the same batch, D_k = A_k - B_k for seeds k = 0, 1. (D_0 - D_1) / 2 contains only
noise and has the same variance as the noise in the two-seed mean (D_0 + D_1) / 2, so the share of pairs
with |(D_0 - D_1) / 2| > bar estimates the two-sided rate at which noise alone clears the bar.

usage: python3 -I tails.py <runs.jsonl> <out.json>
"""
import os, json, sys, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, snr

if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = [p for p in pairs.seed_pairs(runs) if p["hidden"] == 512]
    out = {}
    for label, near in (("all stable configs", None), ("configs within 0.01 of the batch median", 0.01)):
        G = snr.batches(P, near)
        half = {"macro16": [], "macro14_noptb_nowt": []}
        for g in G:
            for (a0, a1), (b0, b1) in itertools.combinations(g, 2):
                for m in half:
                    half[m].append(abs(((a0[m] - b0[m]) - (a1[m] - b1[m])) / 2))
        n = len(half["macro16"])
        rate16 = sum(x > 0.005 for x in half["macro16"]) / n
        sd16 = math.sqrt(sum(x * x for x in half["macro16"]) / n)
        gauss16 = math.erfc(0.005 / sd16 / math.sqrt(2))
        # the 14-subset bar with the same empirical exceedance rate as 0.005 on the full macro
        xs = sorted(half["macro14_noptb_nowt"], reverse=True)
        k = max(1, round(rate16 * n))
        bar14 = xs[k - 1]
        out[label] = {"config_pairs": n, "rate_macro16_over_0.005": rate16, "gaussian_rate_same_sd": gauss16,
                      "rate_macro14_over_0.004": sum(x > 0.004 for x in xs) / n, "macro14_bar_matching_rate": bar14}
        print(f"== {label}: {n} within-batch config pairs")
        print(f"   noise alone clears 0.005 on the full macro in {rate16:.2%} of pairs (Gaussian with the same SD: {gauss16:.2%})")
        print(f"   on the 14-subset macro it clears 0.004 in {out[label]['rate_macro14_over_0.004']:.2%}; the bar with the same rate as 0.005 is {bar14:.4f}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
