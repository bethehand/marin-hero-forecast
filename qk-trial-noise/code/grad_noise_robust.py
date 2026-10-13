"""Grad-norm reproducibility at 16K: per-layer (median |log diff|) vs aggregated over layers by parameter type, reruns vs seeds, excluding the first step."""
import json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "..", "data", "grads_16k_180k.json")))
K, R = G["keys"], G["runs"]
STEPS = [s for s in sorted(R["mhep-ctx16k-s0-20260930"], key=int) if s != "180000"]
SAME = [("mhep-ctx16k-s0-20260930", "ctx9615-ctl-16k-s0-01"), ("ctx9615-ctl-16k-s0-01", "ctx9615-tip-16k-s0-01"),
        ("mhep-ctx16k-s0-20260930", "ctx9615-tip-16k-s0-01")]
SEEDS = [("mhep-ctx16k-s0-20260930", "mhep-ctx16k-s1-20260930"), ("mhep-ctx16k-s0-20260930", "mhep-ctx16k-s2-20260930"),
         ("mhep-ctx16k-s1-20260930", "mhep-ctx16k-s2-20260930")]


def gtype(k):
    n = k[len("grad/norm/"):]
    if not n.startswith("stacked_blocks.") or n.startswith("stacked_blocks.stacked"):
        return None
    return re.sub(r"^stacked_blocks\.\d+\.", "", re.sub(r"\.\d+\.", ".N.", n))


def main():
    types = {}
    for k in K:
        t = gtype(k)
        if t:
            types.setdefault(t, []).append(k)
    top = [k for k in K if gtype(k) is None and not k.startswith("grad/norm/stacked_blocks.stacked")]
    res = {}
    print(f"{'type (48 layers)':32} {'per-layer median |dlog|':>24} {'all-layer total |dlog|':>24}")
    print(f"{'':32} {'reruns':>11} {'seeds':>11} {'reruns':>11} {'seeds':>11}")
    for t, ks in sorted(types.items()):
        def per_layer(pairs):
            return float(np.median([abs(np.log(R[a][s][k]) - np.log(R[b][s][k])) for a, b in pairs for s in STEPS for k in ks]))
        def total(run, s):
            return float(np.sqrt(sum(R[run][s][k] ** 2 for k in ks)))
        def agg(pairs):
            return float(np.median([abs(np.log(total(a, s)) - np.log(total(b, s))) for a, b in pairs for s in STEPS]))
        r = dict(layer_same=per_layer(SAME), layer_seed=per_layer(SEEDS), total_same=agg(SAME), total_seed=agg(SEEDS))
        res[t] = r
        print(f"{t:32} {r['layer_same']*100:10.1f}% {r['layer_seed']*100:10.1f}% {r['total_same']*100:10.1f}% {r['total_seed']*100:10.1f}%")
    print("\ntop-level parameters, median |dlog|:")
    for k in top:
        same = np.median([abs(np.log(R[a][s][k]) - np.log(R[b][s][k])) for a, b in SAME for s in STEPS])
        seed = np.median([abs(np.log(R[a][s][k]) - np.log(R[b][s][k])) for a, b in SEEDS for s in STEPS])
        res[k] = dict(same=float(same), seed=float(seed))
        print(f"  {k[len('grad/norm/'):]:30} reruns {same*100:5.1f}%  seeds {seed*100:5.1f}%")
    json.dump(res, open(os.path.join(HERE, "..", "results", "grad_noise_robust.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
