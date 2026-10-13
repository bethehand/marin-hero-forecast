"""Does the seed fix the batch content across arms? Per-step loss correlations in the Sept 30 mhep-ctx runs."""
import json, os, sys
import numpy as np

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "loss_180k.json")))
STEPS = list(range(180000, 180100))
ARMS = ["4k", "8k", "16k"]


def series(name, key="train/loss"):
    return np.array([D[name][str(s)][key] for s in STEPS], dtype=float)


def detrend(x):
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


def main():
    L = {(a, s): series(f"mhep-ctx{a}-s{s}-20260930") for a in ARMS for s in (0, 1, 2)}
    hero = np.array([D["hero-fa4sm100-nomask-step146k"].get(str(s), {}).get("train/loss", np.nan) for s in STEPS])
    print("hero rows in window:", int(np.isfinite(hero).sum()))
    for s in (0, 1, 2):
        d = L[("4k", s)] - hero
        print(f"4K seed {s} minus hero: mean {np.nanmean(d):+.5f}  max|d| {np.nanmax(np.abs(d)):.5f}  corr {np.corrcoef(L[('4k', s)], hero)[0, 1]:.3f}")
    R = {k: detrend(v) for k, v in L.items()}
    print("\nper-step SD after detrending:", {f"{a}-s{s}": round(float(R[(a, s)].std()), 4) for a in ARMS for s in (0, 1, 2)})
    c = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    same_seed = [c(R[(a, s)], R[(b, s)]) for s in (0, 1, 2) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]]
    same_arm = [c(R[(a, s)], R[(a, t)]) for a in ARMS for s in (0, 1, 2) for t in (0, 1, 2) if s < t]
    diff_both = [c(R[(a, s)], R[(b, t)]) for i, a in enumerate(ARMS) for b in ARMS[i + 1:] for s in (0, 1, 2) for t in (0, 1, 2) if s != t]
    print(f"\ncorr of per-step fluctuations: same seed, different arm: mean {np.mean(same_seed):.3f} (min {np.min(same_seed):.3f}, max {np.max(same_seed):.3f})")
    print(f"                              same arm, different seed: mean {np.mean(same_arm):.3f} (min {np.min(same_arm):.3f}, max {np.max(same_arm):.3f})")
    print(f"                              different arm and seed:   mean {np.mean(diff_both):.3f}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "loss_pairing.json"), "w") as f:
        json.dump({"same_seed_cross_arm_corr": same_seed, "same_arm_cross_seed_corr": same_arm,
                   "per_step_sd_detrended": {f"{a}-s{s}": float(R[(a, s)].std()) for a in ARMS for s in (0, 1, 2)}}, f, indent=1)
    print("\nfirst 8 steps, raw loss:")
    for a in ARMS:
        for s in (0, 1, 2):
            print(f"  {a:>3}-s{s}", " ".join(f"{v:.4f}" for v in L[(a, s)][:8]))
    print("  hero    ", " ".join(f"{v:.4f}" for v in hero[:8]))


if __name__ == "__main__":
    main()
