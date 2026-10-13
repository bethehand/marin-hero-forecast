"""How well each metric's two-seed value estimates the true effect on the full 16-subset macro.

The target is the true (seed-averaged) macro16 effect of a config, centred within its batch. For an estimate
built from metric m, the error variance is var(true_m - true_16) + noise_m / 2, where cross-seed covariances
(metric a under one seed against metric b under the other) give the true-effect terms, because the noise is
independent across seeds. "bias" below is the first term, "noise" the second.

usage: python3 -I target_error.py <runs.jsonl> <out.json>
"""
import os, json, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, snr

TARGET = "macro16"
ESTIMATES = ("macro16", "macro15_noptb", "macro14_noptb_nowt", "uncheatable")


def cross(groups, a, b):
    """Symmetrised within-batch covariance of metric a under one seed with metric b under the other."""
    s, df = 0.0, 0
    for g in groups:
        if len(g) < 2:
            continue
        df += len(g) - 1
        ma0 = sum(x[a] for x, _ in g) / len(g)
        mb0 = sum(x[b] for x, _ in g) / len(g)
        ma1 = sum(y[a] for _, y in g) / len(g)
        mb1 = sum(y[b] for _, y in g) / len(g)
        for x, y in g:
            s += 0.5 * ((x[a] - ma0) * (y[b] - mb1) + (x[b] - mb0) * (y[a] - ma1))
    return s / df


if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = [p for p in pairs.seed_pairs(runs) if p["hidden"] == 512]
    out = {}
    for label, near in (("configs within 0.01 of the batch median", 0.01), ("all stable configs", None)):
        G = snr.batches(P, near)
        st, _ = snr.stats(G, list(ESTIMATES))
        out[label] = {}
        print("==", label)
        for m in ESTIMATES:
            bias_var = cross(G, m, m) + cross(G, TARGET, TARGET) - 2 * cross(G, m, TARGET)
            noise_var = st[m]["noise_var"] / 2
            rmse = math.sqrt(max(bias_var, 0) + noise_var)
            out[label][m] = {"rmse": rmse, "bias_sd": math.sqrt(max(bias_var, 0)), "noise_sd": math.sqrt(noise_var)}
            print(f"   two-seed {m:20s} RMSE {rmse:.5f}  (bias {math.sqrt(max(bias_var, 0)):.5f}, noise {math.sqrt(noise_var):.5f})")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
