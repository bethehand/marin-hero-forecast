"""Signal and noise of candidate metrics across the configs of each Belay batch.

Within a batch, every config A with both seeds gives x_A0 and x_A1 (a metric, centred on the batch mean of
that seed). Pooled over batches: C01 = mean(x_A0 x_A1) estimates the variance of the true config effects,
and N = (V0 + V1)/2 - C01 the per-seed noise variance. SNR = C01 / N; the seed-to-seed correlation is
C01 / sqrt(V0 V1). A metric with higher SNR ranks configs more reproducibly.

usage: python3 -I snr.py <runs.jsonl> <out.json>
"""
import os, json, math, sys, collections, random, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, seed_noise as sn

DROP = {"macro16": (), "macro15_noptb": ("ptb",), "macro14_noptb_nowt": ("ptb", "wikitext_103"),
        "macro13_noptb_nowt_notw": ("ptb", "wikitext_103", "twitterAAE_HELM_fixed")}


def values(r):
    v = {name: sum(x for s, x in r["pal"].items() if s not in d) / (16 - len(d)) for name, d in DROP.items()}
    v["uncheatable"] = sum(r["unc"].values()) / len(r["unc"])
    v["ptb"], v["wikitext_103"] = r["pal"]["ptb"], r["pal"]["wikitext_103"]
    return v


def stats(groups, keys):
    acc = {k: [0.0, 0.0, 0.0] for k in keys}
    df = 0
    for g in groups:
        if len(g) < 2:
            continue
        df += len(g) - 1
        for k in keys:
            m0 = sum(a[k] for a, _ in g) / len(g)
            m1 = sum(b[k] for _, b in g) / len(g)
            for a, b in g:
                x0, x1 = a[k] - m0, b[k] - m1
                acc[k][0] += x0 * x0; acc[k][1] += x1 * x1; acc[k][2] += x0 * x1
    out = {}
    for k, (v0, v1, c) in acc.items():
        v0, v1, c = v0 / df, v1 / df, c / df
        noise = (v0 + v1) / 2 - c
        out[k] = {"signal_var": c, "noise_var": noise, "snr": c / noise if noise > 0 else float("inf"),
                  "seed_corr": c / math.sqrt(v0 * v1)}
    return out, df


def batches(P, near=None, by_metric="macro16"):
    """Seed pairs grouped by batch as (seed-0 values, seed-1 values), unstable pairs removed.
    near: keep only configs whose two-seed value of by_metric is within this distance of their batch median."""
    _, dropped = sn.trimmed(P)
    by = collections.defaultdict(list)
    for p in P:
        if p["name"] not in dropped:
            by[p["batch"]].append((values(p["r0"]), values(p["r1"])))
    groups = list(by.values())
    if near is not None:
        out = []
        for g in groups:
            ms = sorted((a[by_metric] + b[by_metric]) / 2 for a, b in g)
            med = statistics.median(ms)
            out.append([(a, b) for a, b in g if abs((a[by_metric] + b[by_metric]) / 2 - med) <= near])
        groups = out
    return groups


def boot(groups, keys, n=2000, seed=0):
    rng = random.Random(seed)
    res = {k: [] for k in ("snr_ratio_14_vs_16", "snr_ratio_15_vs_16", "noise_sd_ratio_14_vs_16")}
    for _ in range(n):
        s, _ = stats([rng.choice(groups) for _ in groups], keys)
        res["snr_ratio_14_vs_16"].append(s["macro14_noptb_nowt"]["snr"] / s["macro16"]["snr"])
        res["snr_ratio_15_vs_16"].append(s["macro15_noptb"]["snr"] / s["macro16"]["snr"])
        res["noise_sd_ratio_14_vs_16"].append(math.sqrt(s["macro14_noptb_nowt"]["noise_var"] / s["macro16"]["noise_var"]))
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    return {k: [q(v, 0.025), q(v, 0.5), q(v, 0.975)] for k, v in res.items()}


if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = [p for p in pairs.seed_pairs(runs) if p["hidden"] == 512]
    keys = list(DROP) + ["uncheatable", "ptb", "wikitext_103"]
    report = {}
    for label, near, by in (("all stable configs", None, "macro16"),
                            ("configs within 0.02 of the batch median", 0.02, "macro16"),
                            ("configs within 0.01 of the batch median", 0.01, "macro16"),
                            ("configs within 0.01 of the batch median of uncheatable", 0.01, "uncheatable"),
                            ("configs within 0.01 of the batch median of macro14", 0.01, "macro14_noptb_nowt")):
        groups = batches(P, near, by)
        s, df = stats(groups, keys)
        b = boot([g for g in groups if len(g) >= 2], keys)
        report[label] = {"df": df, "configs": sum(len(g) for g in groups if len(g) >= 2), "stats": s, "bootstrap_95": b}
        print(f"== {label}: {report[label]['configs']} configs in batches, df {df}")
        print("  %-26s %11s %11s %7s %9s" % ("metric", "signal SD", "noise SD", "SNR", "seed corr"))
        for k in keys:
            x = s[k]
            print("  %-26s %11.5f %11.5f %7.2f %9.3f" % (k, math.sqrt(max(x["signal_var"], 0)), math.sqrt(x["noise_var"]), x["snr"], x["seed_corr"]))
        print("  bootstrap over batches (2.5%, 50%, 97.5%):", {k: [round(v, 3) for v in vs] for k, vs in b.items()})
    json.dump(report, open(sys.argv[2], "w"), indent=1)
