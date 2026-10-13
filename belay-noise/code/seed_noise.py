"""How much of the seed noise in Belay's paired comparisons comes from each Paloma subset.

For a seed pair (same config, seeds 0 and 1) the seed difference is e = L(seed 0) - L(seed 1), per subset and
for each macro. Within one batch, the spread of e across configs is the noise of a single-seed paired delta:
if L(A, k) = mu_A + beta_k + eps_{A,k}, then e_A - mean(e) = eps_{A,0} - eps_{A,1} - mean(...), and a one-seed
paired delta A - B carries eps_{A,k} - eps_{B,k}, with the same variance 2 var(eps). The batch mean removes
beta_0 - beta_1, which paired deltas cancel anyway.

Share of macro noise from subset s: cov(e_s, e_macro) / (16 var(e_macro)); the 16 shares sum to 1.

usage: python3 -I seed_noise.py <runs.jsonl> <out.json>
"""
import os, json, math, sys, collections, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs

DIRTY = ("ptb", "wikitext_103")


def metrics(r):
    pal, unc = r["pal"], r["unc"]
    m = {f"pal/{s}": v for s, v in pal.items()}
    m.update({f"unc/{s}": v for s, v in unc.items()})
    m["macro16"] = sum(pal.values()) / len(pal)
    clean = [v for s, v in pal.items() if s not in DIRTY]
    m["macro14"] = sum(clean) / len(clean)
    m["unc_macro"] = sum(unc.values()) / len(unc)
    return m


def within_batch(rows, keys):
    """rows: list of (batch, {key: e}). Demean e within each batch with >= 2 rows; return deviations and df."""
    by = collections.defaultdict(list)
    for b, e in rows:
        by[b].append(e)
    dev, df = [], 0
    for b, es in by.items():
        if len(es) < 2:
            continue
        mean = {k: sum(e[k] for e in es) / len(es) for k in keys}
        dev += [{k: e[k] - mean[k] for k in keys} for e in es]
        df += len(es) - 1
    return dev, df


def cov(dev, df, a, b):
    return sum(d[a] * d[b] for d in dev) / df


TRIM = 0.01  # drop pairs whose macro16 seed difference is this far from their batch median (unstable runs)


def trimmed(P):
    """Seed pairs minus unstable ones; returns (kept rows, dropped names)."""
    rows = []
    for p in P:
        m0, m1 = metrics(p["r0"]), metrics(p["r1"])
        rows.append((p, {k: m0[k] - m1[k] for k in m0}))
    by = collections.defaultdict(list)
    for p, e in rows:
        by[p["batch"]].append(e["macro16"])
    med = {b: statistics.median(v) for b, v in by.items()}
    keep = [(p["batch"], e) for p, e in rows if len(by[p["batch"]]) < 2 or abs(e["macro16"] - med[p["batch"]]) <= TRIM]
    dropped = [p["name"] for p, e in rows if len(by[p["batch"]]) >= 2 and abs(e["macro16"] - med[p["batch"]]) > TRIM]
    return keep, dropped


def analyse(P, label):
    rows, dropped = trimmed(P)
    keys = list(rows[0][1])
    dev, df = within_batch(rows, keys)
    sd = {k: math.sqrt(cov(dev, df, k, k)) for k in keys}
    v16 = cov(dev, df, "macro16", "macro16")
    pal_keys = [k for k in keys if k.startswith("pal/")]
    share = {k[4:]: cov(dev, df, k, "macro16") / (16 * v16) for k in pal_keys}
    corr = {k[4:]: cov(dev, df, k, "macro16") / (sd[k] * sd["macro16"]) for k in pal_keys}
    out = {"label": label, "pairs": len(rows), "pairs_used": len(dev), "df": df, "dropped_unstable": dropped,
           "sd_single_seed_paired_delta": {k: sd[k] for k in ("macro16", "macro14", "unc_macro")},
           "subset_sd": {k[4:]: sd[k] for k in pal_keys}, "share_of_macro16_noise_var": share,
           "corr_with_macro16": corr,
           "share_ptb_plus_wikitext": share["ptb"] + share["wikitext_103"]}
    # sanity: the mean of e over all rows (seed 0 minus seed 1), i.e. the common seed effect
    out["mean_seed_effect"] = {k: sum(e[k] for _, e in rows) / len(rows) for k in ("macro16", "macro14", "unc_macro")}
    return out


def show(o):
    print(f"== {o['label']}: {o['pairs']} pairs kept ({len(o['dropped_unstable'])} unstable dropped), {o['pairs_used']} in batches with >= 2 pairs, df {o['df']}")
    s = o["sd_single_seed_paired_delta"]
    print("  SD of a one-seed paired delta: macro16 %.5f | macro14 %.5f (%.0f%%) | uncheatable %.5f" % (
        s["macro16"], s["macro14"], 100 * (s["macro14"] / s["macro16"] - 1), s["unc_macro"]))
    print("  share of macro16 noise variance from ptb + wikitext_103: %.1f%%" % (100 * o["share_ptb_plus_wikitext"]))
    print("  %-34s %9s %9s %7s" % ("subset", "SD(e_s)", "share", "corr"))
    for k in sorted(o["subset_sd"], key=lambda k: -o["share_of_macro16_noise_var"][k]):
        print("  %-34s %9.5f %8.1f%% %7.2f" % (k, o["subset_sd"][k], 100 * o["share_of_macro16_noise_var"][k], o["corr_with_macro16"][k]))


if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = pairs.seed_pairs(runs)
    d512 = [p for p in P if p["hidden"] == 512]
    res = [analyse(d512, "d512 strict pairs, all step counts"),
           analyse([p for p in d512 if p["steps"] == 2817], "d512 strict pairs, 2817 steps"),
           analyse([p for p in d512 if p["created"] < "2026-09-29"], "d512 strict pairs, created before 09-29"),
           analyse([p for p in d512 if p["created"] >= "2026-09-29"], "d512 strict pairs, created 09-29 or later")]
    Q = [q for q in pairs.config_pairs(runs) if q["hidden"] == 512]
    res.append(analyse(Q, "d512 config pairs (looser matching)"))
    for o in res:
        show(o)
    json.dump(res, open(sys.argv[2], "w"), indent=1)
