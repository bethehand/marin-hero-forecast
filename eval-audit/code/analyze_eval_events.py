#!/usr/bin/env python3
"""Noise attribution and drop-and-recover events in the hero's eval curves.

Uses only public W&B data (the frozen snapshot in ../../data).

Rules fixed before looking at the results:
  - Noise attribution, after the data swap (from step 110,999) and separately before it.
    Three detrending methods: residuals from a quadratic in log(step), first differences,
    and residuals from a centred 5-point rolling median. A subset's share of the macro's
    variance is Cov(r_s / 16, m) / Var(m), where m is the mean of the 16 subset residuals;
    the shares add up to 1. Block bootstrap (block length 3) for the main method.
  - Event detection, the same rule for all 23 subsets, split at the step-108,000 swap:
      baseline = Theil-Sen line through the previous 6 unflagged evals, extrapolated to the
      current eval; residual r = value - baseline;
      sigma = 1.4826 x MAD of the first-pass residuals (no points excluded);
      drop: r < -z sigma (main rule z = 5; z = 4 and 6 also reported); adjacent drops merge
      into one event;
      recovery: within 5 evals after the event, some eval has r > -1.5 sigma against the same
      baseline; otherwise "not recovered".
  - Controls: the confirmed PTB contamination (steps 206,999-212,999) must be detected; the 7
    Uncheatable subsets and the large Paloma subsets are negative controls for false alarms.

A second detector added after seeing the results (two-sided spike statistic):
  The first detector extrapolates from the previous 6 evals. In series that drop often, like
  PTB and WikiText, unflagged drops enter the baseline and inflate sigma; it missed even the
  confirmed 206,999 and 212,999 evals. So we add a statistic that needs no extrapolation:
      d_t = L_t - (L_{t-1} + L_{t+1}) / 2, sigma = 1.4826 x MAD(d), z = d / sigma.
  A smooth trend cancels in d, and a one-eval drop stands out. Before the swap, the fast
  initial descent (steps < 20k) is skipped. Points with |z| >= 4 and >= 6 are reported, with
  downward and upward counts separately (contamination can only produce downward points).

Usage:
  python analyze_eval_events.py <snapshot dir> <output dir>
"""

import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

LINEAGE = [  # the hero's W&B runs in hand-off order; when a step is logged twice, the later run wins
    "hero-84579c80-072503", "hero-12d8b6f0-dee637", "hero-wd-gate-router-p02-step58k",
    "hero-ragged_a2a-nccl2307-ep-step81k", "hero-nopdl-step108k", "hero-main-step121638",
    "hero-fa4sm100-nomask-step146k",
]
SWAP_STEP = 108_000  # data-mix swap
PAL = re.compile(r"^eval_dropless/paloma/(.+)-llama3/loss$")
UNC = re.compile(r"^eval_dropless/uncheatable_eval/(.+)-llama3/loss$")
W_BASE, Z_MAIN, Z_ALT, Z_REC, K_REC = 6, 5.0, (4.0, 6.0), 1.5, 5
KNOWN_PTB_CONTAM = (206_999, 209_999, 212_999)


def load(snap):
    l = pd.read_csv(os.path.join(snap, "tables", "wandb_long.csv"), usecols=["category", "run", "key", "step", "value"])
    h = l[(l.category == "hero") & (l.step > 0)].copy()
    h["ord"] = h.run.map({r: i for i, r in enumerate(LINEAGE)})
    h = h.sort_values(["step", "key", "ord"]).drop_duplicates(["step", "key"], keep="last")
    w = h.pivot_table(index="step", columns="key", values="value").sort_index()
    pal = {PAL.match(k).group(1): k for k in w.columns if PAL.match(k)}
    unc = {UNC.match(k).group(1): k for k in w.columns if UNC.match(k)}
    P = w[[pal[s] for s in sorted(pal)]].set_axis(sorted(pal), axis=1)
    U = w[[unc[s] for s in sorted(unc)]].set_axis(sorted(unc), axis=1)
    macro = w["eval_dropless/paloma/macro_loss"]
    return P, U, macro


# ---------------- noise attribution ----------------
def residuals(df, method):
    if method == "diff":
        return df.diff().dropna()
    if method == "rollmed":
        return (df - df.rolling(5, center=True, min_periods=3).median()).dropna()
    x = np.log(df.index.values.astype(float))
    out = {}
    for c in df.columns:
        coef = np.polyfit(x, df[c].values, 2)
        out[c] = df[c].values - np.polyval(coef, x)
    return pd.DataFrame(out, index=df.index)


def shares(R):
    m = R.mean(axis=1)
    v = m.var()
    return {c: float(np.cov(R[c] / R.shape[1], m)[0, 1] / v) for c in R.columns}, float(m.std())


def attribution(P, lo, hi, n_boot=2000, seed=0):
    sub = P[(P.index >= lo) & (P.index <= hi)]
    out = {"steps": [int(sub.index.min()), int(sub.index.max())], "n_evals": int(len(sub)), "methods": {}}
    clean = [c for c in sub.columns if c not in ("ptb", "wikitext_103")]
    for method in ("quad_log", "diff", "rollmed"):
        R = residuals(sub, method)
        sh, sd16 = shares(R)
        sd14 = float(R[clean].mean(axis=1).std())
        top = dict(sorted(sh.items(), key=lambda kv: -kv[1])[:5])
        out["methods"][method] = {"sd_macro16": sd16, "sd_macro14_without_ptb_wikitext": sd14,
                                  "sd_ratio_14_over_16": sd14 / sd16, "share_ptb": sh["ptb"],
                                  "share_wikitext_103": sh["wikitext_103"], "top5": top, "all": sh}
    # block bootstrap (main method)
    rng = np.random.default_rng(seed)
    R = residuals(sub, "quad_log").values
    T, b = R.shape[0], 3
    cols = list(sub.columns)
    ip, iw = cols.index("ptb"), cols.index("wikitext_103")
    bs = []
    for _ in range(n_boot):
        starts = rng.integers(0, T - b + 1, size=int(np.ceil(T / b)))
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:T]
        Rb = R[idx]
        m = Rb.mean(axis=1)
        v = m.var(ddof=1)
        cov = lambda j: np.cov(Rb[:, j] / R.shape[1], m)[0, 1] / v
        sd14 = Rb[:, [j for j in range(len(cols)) if j not in (ip, iw)]].mean(axis=1).std(ddof=1)
        bs.append((cov(ip), cov(iw), sd14 / m.std(ddof=1)))
    bs = np.array(bs)
    out["bootstrap_quad_log_95ci"] = {"share_ptb": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                      "share_wikitext_103": np.percentile(bs[:, 1], [2.5, 97.5]).tolist(),
                                      "sd_ratio_14_over_16": np.percentile(bs[:, 2], [2.5, 97.5]).tolist()}
    return out


# ---------------- event detection ----------------
def baseline_at(steps, vals, i, usable):
    prev = [j for j in range(i) if usable[j]][-W_BASE:]
    if len(prev) < W_BASE:
        return None
    x, y = steps[prev], vals[prev]
    slope, inter, _, _ = theilslopes(y, x)
    return slope, inter


def detect(series, z):
    """Events in one subset and one mix segment. Returns (events, residual per eval, sigma)."""
    steps = series.index.values.astype(float)
    vals = series.values.astype(float)
    n = len(vals)
    # first pass: no points excluded, estimate sigma
    r0 = []
    for i in range(n):
        bl = baseline_at(steps, vals, i, [True] * n)
        if bl is not None:
            r0.append(vals[i] - (bl[0] * steps[i] + bl[1]))
    r0 = np.array(r0)
    sigma = 1.4826 * np.median(np.abs(r0 - np.median(r0))) if len(r0) else np.nan
    # second pass, in time order: flagged evals are kept out of later baselines
    usable = [True] * n
    resid = np.full(n, np.nan)
    flags = np.zeros(n, bool)
    lines = [None] * n
    for i in range(n):
        bl = baseline_at(steps, vals, i, usable)
        if bl is None:
            continue
        lines[i] = bl
        resid[i] = vals[i] - (bl[0] * steps[i] + bl[1])
        if resid[i] < -z * sigma:
            flags[i] = True
            usable[i] = False
    events = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        slope, inter = lines[i]  # baseline at the start of the event
        rec = None
        for k in range(j + 1, min(n, j + 1 + K_REC)):
            if vals[k] - (slope * steps[k] + inter) > -Z_REC * sigma:
                rec = k - j
                break
        depth = float(np.min(resid[i:j + 1]))
        events.append({"first_step": int(steps[i]), "last_step": int(steps[j]), "n_evals": j - i + 1,
                       "depth_nats": depth, "depth_sigma": depth / sigma, "recovered_after_evals": rec,
                       "value_at_min": float(vals[i + int(np.argmin(resid[i:j + 1]))])})
        i = j + 1
    return events, resid, sigma


def run_events(df, family):
    rows, zmap, sig = [], {}, {}
    for c in df.columns:
        s = df[c].dropna()
        for seg_name, seg in (("pre_swap", s[s.index < SWAP_STEP]), ("post_swap", s[s.index >= SWAP_STEP])):
            for z in (Z_MAIN,) + Z_ALT:
                ev, resid, sigma = detect(seg, z)
                for e in ev:
                    rows.append(dict(family=family, subset=c, segment=seg_name, z=z, sigma=sigma, **e))
                if z == Z_MAIN:
                    zmap[(c, seg_name)] = pd.Series(resid / sigma, index=seg.index)
                    sig[(c, seg_name)] = sigma
    return pd.DataFrame(rows), zmap, sig


def spike_events(df, family):
    rows, summ = [], []
    for c in df.columns:
        for seg_name, seg in (("pre_swap", df[c][df.index < SWAP_STEP]), ("post_swap", df[c][df.index >= SWAP_STEP])):
            s = seg.dropna()
            if seg_name == "pre_swap":
                s = s[s.index >= 20_000]
            d = (s - (s.shift(1) + s.shift(-1)) / 2).dropna()
            sd = 1.4826 * np.median(np.abs(d - np.median(d)))
            z = d / sd
            summ.append({"family": family, "subset": c, "segment": seg_name, "sigma_spike": sd, "n": int(len(z)),
                         "dips_le_-4": int((z <= -4).sum()), "spikes_ge_+4": int((z >= 4).sum()),
                         "dips_le_-6": int((z <= -6).sum()), "spikes_ge_+6": int((z >= 6).sum()),
                         "min_z": float(z.min()), "max_z": float(z.max())})
            for st, zz in z[(z <= -4) | (z >= 4)].items():
                rows.append({"family": family, "subset": c, "segment": seg_name, "step": int(st), "z": float(zz),
                             "d_nats": float(d[st]), "value": float(s[st])})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def main(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    P, U, macro = load(snap)
    check = float((P.mean(axis=1) - macro).abs().max())
    res = {"data": {"paloma_subsets": list(P.columns), "uncheatable_subsets": list(U.columns),
                    "steps": [int(P.index.min()), int(P.index.max())], "n_evals": int(len(P)),
                    "max_abs_diff_mean16_vs_logged_macro": check}}
    res["attribution_post_swap"] = attribution(P, 110_999, 10 ** 9)
    res["attribution_pre_swap"] = attribution(P, 1, 107_999)

    ev_p, zp, sp = run_events(P, "paloma")
    ev_u, zu, su = run_events(U, "uncheatable")
    ev = pd.concat([ev_p, ev_u], ignore_index=True)
    ev.to_csv(os.path.join(outdir, "events_all_z.csv"), index=False)
    main_ev = ev[ev.z == Z_MAIN].copy()
    main_ev["macro_effect_nats"] = np.where(main_ev.family == "paloma", main_ev.depth_nats / 16, main_ev.depth_nats / len(U.columns))
    main_ev.to_csv(os.path.join(outdir, "events_z5.csv"), index=False)

    def count(df, fam):
        d = df[df.family == fam]
        return {z: int((d.z == z).sum()) for z in (4.0, 5.0, 6.0)}
    ctrl = {"known_ptb_contamination_detected_z5": bool(((main_ev.subset == "ptb") & (main_ev.first_step <= 212_999)
                                                         & (main_ev.last_step >= 206_999)).any()),
            "events_by_family": {"paloma": count(ev, "paloma"), "uncheatable": count(ev, "uncheatable")},
            "events_by_subset_z5": {f"{f}/{c}": int(n) for (f, c), n in main_ev.groupby(["family", "subset"]).size().items()}}
    res["events_controls"] = {str(k): v for k, v in ctrl.items()}
    res["events_z5"] = main_ev.sort_values(["subset", "first_step"]).to_dict("records")
    sp_p, sm_p = spike_events(P, "paloma")
    sp_u, sm_u = spike_events(U, "uncheatable")
    spikes = pd.concat([sp_p, sp_u], ignore_index=True)
    spike_summary = pd.concat([sm_p, sm_u], ignore_index=True)
    spikes.to_csv(os.path.join(outdir, "spike_points_abs_z_ge_4.csv"), index=False)
    spike_summary.to_csv(os.path.join(outdir, "spike_summary_by_subset.csv"), index=False)
    res["spike_points_abs_z_ge_4"] = spikes.sort_values(["subset", "step"]).to_dict("records")
    res["spike_totals"] = {fam: {"dips_le_-4": int(g["dips_le_-4"].sum()), "spikes_ge_+4": int(g["spikes_ge_+4"].sum())}
                           for fam, g in spike_summary.groupby("family")}
    json.dump(res, open(os.path.join(outdir, "phase1_results.json"), "w"), indent=1, default=str)

    # figure 1: residual z-scores and events for all 23 subsets
    fams = [("paloma", P, zp), ("uncheatable", U, zu)]
    allc = [(f, c) for f, df, _ in fams for c in df.columns]
    nc = 4
    nr = int(np.ceil(len(allc) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(18, 2.3 * nr), sharex=True)
    for ax, (f, c) in zip(axes.flat, allc):
        zm = zp if f == "paloma" else zu
        for seg in ("pre_swap", "post_swap"):
            zs = zm.get((c, seg))
            if zs is not None:
                ax.plot(zs.index / 1000, zs.values, lw=0.9, color="black")
        e = main_ev[(main_ev.family == f) & (main_ev.subset == c)]
        for _, r in e.iterrows():
            ax.axvspan(r.first_step / 1000 - 1.5, r.last_step / 1000 + 1.5, color="crimson", alpha=0.3)
        ax.axhline(-Z_MAIN, color="crimson", lw=0.6, ls="--")
        ax.axvline(SWAP_STEP / 1000, color="grey", lw=0.6, ls=":")
        ax.set_ylim(-25, 8)
        ax.set_title(f"{f}: {c}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(allc):]:
        ax.axis("off")
    fig.supxlabel("step (thousands)")
    fig.supylabel("residual vs. local baseline (σ units)")
    fig.suptitle("Hero eval subsets: drops below a local baseline (red = event at 5σ; dotted = data swap)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_events_all_subsets.png"), dpi=110)

    # figure 2: raw PTB and WikiText-103 loss
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, c in zip(axes, ("ptb", "wikitext_103")):
        ax.plot(P.index / 1000, P[c], "-o", ms=2.5, lw=1, color="black")
        e = main_ev[(main_ev.family == "paloma") & (main_ev.subset == c)]
        for _, r in e.iterrows():
            ax.axvspan(r.first_step / 1000 - 1.5, r.last_step / 1000 + 1.5, color="crimson", alpha=0.25)
        for s in KNOWN_PTB_CONTAM:
            ax.axvline(s / 1000, color="royalblue", lw=0.6, ls="--")
        ax.axvline(SWAP_STEP / 1000, color="grey", lw=0.6, ls=":")
        ax.set_ylabel(f"{c} loss")
        ax.grid(alpha=0.3)
    axes[0].set_title("PTB and WikiText-103 (Paloma) on the hero: red = detected drop events; blue dashed = known PTB contamination evals")
    axes[1].set_xlabel("step (thousands)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_ptb_wikitext.png"), dpi=120)

    # console summary
    print("macro check: max |mean of 16 subsets - logged macro|", check)
    for k in ("attribution_post_swap", "attribution_pre_swap"):
        a = res[k]
        print(f"\n== {k} {a['steps']} n={a['n_evals']}")
        for m, v in a["methods"].items():
            print(f"  {m:8s} SD16 {v['sd_macro16']:.4f}  SD14 {v['sd_macro14_without_ptb_wikitext']:.4f}  "
                  f"ratio {v['sd_ratio_14_over_16']:.2f}  PTB {v['share_ptb']:.0%}  Wiki {v['share_wikitext_103']:.0%}  "
                  f"top5 {{{', '.join(f'{c}: {s:.0%}' for c, s in v['top5'].items())}}}")
        print("  bootstrap 95%:", {kk: [round(x, 2) for x in vv] for kk, vv in a["bootstrap_quad_log_95ci"].items()})
    print("\n== events by z", res["events_controls"]["events_by_family"])
    print("known PTB contamination detected (z=5):", res["events_controls"]["known_ptb_contamination_detected_z5"])
    print(main_ev[["family", "subset", "segment", "first_step", "last_step", "n_evals", "depth_nats", "depth_sigma",
                   "recovered_after_evals", "macro_effect_nats"]].round(4).to_string())
    print("\n== two-sided spike statistic: points with |z| >= 4 ==")
    print(spikes.sort_values(["family", "subset", "step"]).round(4).to_string(index=False))
    print("totals:", res["spike_totals"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
