"""Per-step loss noise: same seed and packing (reruns) vs different seeds, from the 180k restore runs."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "..", "data", "loss_180k.json")))
B = json.load(open(os.path.join(HERE, "..", "data", "loss_reruns_180k.json")))
D = {**A, **B}
STEPS = list(range(180000, 180100))
ARMS = ["4k", "8k", "16k"]


def s(name, key="train/loss"):
    return np.array([D[name][str(t)][key] for t in STEPS], dtype=float)


def main():
    out = {}
    for a in ARMS:
        sept = {k: s(f"mhep-ctx{a}-s{k}-20260930") for k in (0, 1, 2)}
        ctl, tip = s(f"ctx9615-ctl-{a}-s0-01"), s(f"ctx9615-tip-{a}-s0-01")
        same = {"sept-s0 vs ctl": sept[0] - ctl, "ctl vs tip": ctl - tip, "sept-s0 vs tip": sept[0] - tip}
        diff_seed = [sept[i] - sept[j] for i in (0, 1, 2) for j in (0, 1, 2) if i < j]
        print(f"== {a}")
        for k, d in same.items():
            print(f"  same seed, {k:15}: per-step SD {d.std():.5f}  max|d| {np.abs(d).max():.5f}  mean {d.mean():+.5f}  corr {np.corrcoef(*(sept[0], ctl) if 'ctl' in k and 'sept' in k else (ctl, tip))[0,1]:.4f}")
        sd_diff_seed = np.mean([d.std() for d in diff_seed])
        print(f"  different seeds           : per-step SD of difference {sd_diff_seed:.4f}")
        # error of a window-mean comparison, last 50 steps
        w = slice(50, 100)
        m_same = [abs(d[w].mean()) for d in same.values()]
        m_seed = [abs(d[w].mean()) for d in diff_seed]
        print(f"  |mean diff| over the last 50 steps: same seed {np.round(m_same, 5)}, different seeds {np.round(m_seed, 4)}")
        first = [float(abs(x[0] - y[0])) for x, y in ((sept[0], ctl), (ctl, tip), (sept[0], tip))]
        first_seed = [float(abs(sept[i][0] - sept[j][0])) for i in (0, 1, 2) for j in (0, 1, 2) if i < j]
        worst = int(np.argmax(np.max([np.abs(d) for d in same.values()], axis=0)))
        sd_wo = [float(np.delete(d, worst).std()) for d in same.values()]
        print(f"  first step: same seed |d| {first}, different seeds |d| {np.round(first_seed, 3)}; worst step {STEPS[worst]}, SD without it {np.round(sd_wo, 5)}")
        out[a] = {"first_step_abs_diff_same": first, "first_step_abs_diff_seeds": first_seed,
                  "worst_step": STEPS[worst], "same_seed_step_sd_without_worst": sd_wo,
                  "same_seed_step_sd": {k: float(d.std()) for k, d in same.items()},
                  "same_seed_max_abs": {k: float(np.abs(d).max()) for k, d in same.items()},
                  "diff_seed_step_sd": float(sd_diff_seed),
                  "last50_abs_mean_diff_same": [float(x) for x in m_same],
                  "last50_abs_mean_diff_seeds": [float(x) for x in m_seed]}
    json.dump(out, open(os.path.join(HERE, "..", "results", "paired_noise.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
