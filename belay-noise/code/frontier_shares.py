"""Each Paloma subset's share of the macro's per-seed noise variance, among configs near their batch's frontier.

Within a batch, centre every metric per seed. For metrics a and b, the same-seed covariance minus the
cross-seed covariance is their noise covariance (the true config effects are shared across seeds, the noise
is not). Subset s's share of the macro's noise variance is noisecov(s, macro) / (16 noisevar(macro)).
Unlike seed_noise.py, this uses only configs whose two-seed macro is within 0.01 of their batch median, so
it barely depends on how unstable pairs are handled; it is reported for three trimming thresholds.

usage: python3 -I frontier_shares.py <runs.jsonl> <out.json>
"""
import os, json, sys, collections, statistics, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, seed_noise as sn, snr

NEAR = 0.01


def groups(P, near):
    _, dropped = sn.trimmed(P)
    by = collections.defaultdict(list)
    for p in P:
        if p["name"] in dropped:
            continue
        v0, v1 = snr.values(p["r0"]), snr.values(p["r1"])
        v0.update({"pal/" + s: x for s, x in p["r0"]["pal"].items()})
        v1.update({"pal/" + s: x for s, x in p["r1"]["pal"].items()})
        by[p["batch"]].append((v0, v1))
    out = []
    for g in by.values():
        ms = sorted((a["macro16"] + b["macro16"]) / 2 for a, b in g)
        med = statistics.median(ms)
        g = [(a, b) for a, b in g if abs((a["macro16"] + b["macro16"]) / 2 - med) <= near]
        if len(g) >= 2:
            out.append(g)
    return out


def noise_cov(G, a, b):
    same = cross = 0.0
    df = sum(len(g) - 1 for g in G)
    for g in G:
        ma0 = sum(x[a] for x, _ in g) / len(g); ma1 = sum(y[a] for _, y in g) / len(g)
        mb0 = sum(x[b] for x, _ in g) / len(g); mb1 = sum(y[b] for _, y in g) / len(g)
        for x, y in g:
            same += 0.5 * ((x[a] - ma0) * (x[b] - mb0) + (y[a] - ma1) * (y[b] - mb1))
            cross += 0.5 * ((x[a] - ma0) * (y[b] - mb1) + (y[a] - ma1) * (x[b] - mb0))
    return (same - cross) / df, df


if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = [p for p in pairs.seed_pairs(runs) if p["hidden"] == 512]
    subsets = sorted(P[0]["r0"]["pal"])
    out = {}
    for trim in (0.01, 0.05, None):
        sn.TRIM = trim if trim is not None else float("inf")
        G = groups(P, NEAR)
        n16, df = noise_cov(G, "macro16", "macro16")
        shares = {s: noise_cov(G, "pal/" + s, "macro16")[0] / (16 * n16) for s in subsets}
        label = f"trim {trim}" if trim is not None else "no trimming"
        # 95% intervals from 2,000 bootstrap resamples of batches
        rng = random.Random(0)
        bs = {"ptb": [], "ptb_plus_wikitext": []}
        for _ in range(2000):
            R = [rng.choice(G) for _ in G]
            n = noise_cov(R, "macro16", "macro16")[0]
            sp = noise_cov(R, "pal/ptb", "macro16")[0] / (16 * n)
            sw = noise_cov(R, "pal/wikitext_103", "macro16")[0] / (16 * n)
            bs["ptb"].append(sp); bs["ptb_plus_wikitext"].append(sp + sw)
        ci = {k: [sorted(v)[int(0.025 * (len(v) - 1))], sorted(v)[int(0.975 * (len(v) - 1))]] for k, v in bs.items()}
        out[label] = {"df": df, "configs": sum(len(g) for g in G), "shares": shares,
                      "ptb_plus_wikitext": shares["ptb"] + shares["wikitext_103"], "bootstrap_95": ci}
        print(f"{label:12s} configs {out[label]['configs']} df {df} | ptb {shares['ptb']:.3f} | wikitext_103 {shares['wikitext_103']:.3f}"
              f" | both {out[label]['ptb_plus_wikitext']:.3f} | twitterAAE {shares['twitterAAE_HELM_fixed']:.3f}"
              f" | 95% ptb {ci['ptb'][0]:.3f}-{ci['ptb'][1]:.3f}, both {ci['ptb_plus_wikitext'][0]:.3f}-{ci['ptb_plus_wikitext'][1]:.3f}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
