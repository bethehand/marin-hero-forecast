"""Write the per-run numbers behind the analysis: one row per run of each d512 seed pair.

usage: python3 -I export_pairs.py <runs.jsonl> <out.csv>
"""
import os, csv, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, seed_noise as sn

if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    P = [p for p in pairs.seed_pairs(runs) if p["hidden"] == 512]
    _, dropped = sn.trimmed(P)
    pal = sorted(P[0]["r0"]["pal"]); unc = sorted(P[0]["r0"]["unc"])
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "batch", "run", "seed", "num_train_steps", "created", "unstable_dropped", "paloma_macro"]
                   + [f"paloma/{s}" for s in pal] + ["uncheatable_macro"] + [f"uncheatable/{s}" for s in unc])
        for p in sorted(P, key=lambda p: p["created"]):
            for r in (p["r0"], p["r1"]):
                w.writerow([p["name"], p["batch"], r["name"], r["cfg"][pairs.SEED], p["steps"], r["createdAt"],
                            int(p["name"] in dropped), f"{sum(r['pal'].values()) / 16:.6f}"]
                           + [f"{r['pal'][s]:.6f}" for s in pal] + [f"{sum(r['unc'].values()) / len(r['unc']):.6f}"]
                           + [f"{r['unc'][s]:.6f}" for s in unc])
    print(len(P), "pairs ->", sys.argv[2])
