"""67B qk test (TPU, seed 0, shared batches): per-step paired train-loss differences between arms."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "..", "data", "qk_arms_67b.json")))
P = "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k_yarn_"
ARMS = {"1.3": "mscale00_test20", "1.570": "mscale01_test20", "1.841": "mscale02_test20",
        "long x1.208": "long_mscale01_test20", "long x1.416": "long_mscale02_test20"}
STEPS = list(range(39000, 39020))


def loss(arm):
    rows = D[P + ARMS[arm]]
    return np.array([rows[str(s)]["train/loss"] for s in STEPS], dtype=float)


def main():
    L = {a: loss(a) for a in ARMS}
    base = L["1.3"]
    print("train loss per step, 1.3 arm:", " ".join(f"{v:.4f}" for v in base))
    tt = np.arange(len(base))
    print("1.3 arm loss SD around its linear trend:", round(float((base - np.polyval(np.polyfit(tt, base, 1), tt)).std()), 4))
    print("raw per-step SD across steps (batch to batch), 1.3 arm:", round(float(np.std(np.diff(base))) / np.sqrt(2), 4))
    out = {}
    for a in ARMS:
        if a == "1.3":
            continue
        d = L[a] - base
        print(f"\n{a} minus 1.3, per step: " + " ".join(f"{x*1e3:+.1f}" for x in d) + "  (x1e-3)")
        print(f"   first 5 mean {d[:5].mean()*1e3:+.2f}e-3, last 10 mean {d[10:].mean()*1e3:+.2f}e-3, steps below 0: {(d < 0).sum()}/20")
        print(f"   from step 39005: range {d[5:].min()*1e3:+.1f} to {d[5:].max()*1e3:+.1f} (x1e-3), drift {(d[5:].max()-d[5:].min())*1e3:.1f}e-3, SD {d[5:].std()*1e3:.2f}e-3")
        out[a] = {"per_step_minus_1.3": d.tolist(), "last10_mean": float(d[10:].mean()), "n_below_zero": int((d < 0).sum()),
                  "from_39005_min": float(d[5:].min()), "from_39005_max": float(d[5:].max()), "from_39005_sd": float(d[5:].std())}
    # ranking by mean paired difference over the last 10 steps vs the eval ranking at 39019
    rank = sorted(ARMS, key=lambda a: float((L[a] - base)[10:].mean()))
    ev = {a: (D[P + ARMS[a]]["39019"]["eval/paloma/macro_loss"], D[P + ARMS[a]]["39019"]["eval/uncheatable_eval/macro_loss"]) for a in ARMS}
    import itertools
    rs = [float(np.corrcoef(L[x], L[y])[0, 1]) for x, y in itertools.combinations(ARMS, 2)]
    print(f"\nper-step loss correlation between arms: {min(rs):.3f} to {max(rs):.3f}")
    out["arm_loss_corr_range"] = [min(rs), max(rs)]
    print("rank by paired train-loss diff (last 10 steps):", rank)
    print("rank by Paloma macro at 39019:", sorted(ARMS, key=lambda a: ev[a][0]))
    print("rank by Uncheatable at 39019:", sorted(ARMS, key=lambda a: ev[a][1]))
    json.dump(out, open(os.path.join(HERE, "..", "results", "paired_67b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
