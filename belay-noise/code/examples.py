"""Two fixed-time keep/drop calls from the Belay thread, plus a cluster check, broken down by metric.

usage: python3 -I examples.py <runs.jsonl> <out.json>
"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs, snr

CALLS = {
    "batch 174: candidate 11 vs candidate 10, fixed time (2698 vs 2607 steps; same math, 3.5% faster)":
        (("lc1-b174-ft-kda", "lc1-b174-ft-kda-s1"), ("lc1-b171-ft-c10", "lc1-b171-ft-c10-s1")),
    "batch 200: candidate 12 vs candidate 11, fixed time (2526 vs 2698 steps; cw-rno2a vs cw-us-east-02a)":
        (("lc1-b200-c12-ft", "lc1-b200-c12-ft-s1"), ("lc1-b174-ft-kda", "lc1-b174-ft-kda-s1")),
    "candidate 11 at equal steps on the two clusters (cw-rno2a minus cw-us-east-02a; seed 0 from batch 181, seed 1 from batch 185)":
        (("lc1-b181-ctrl", "lc1-b185-ctrl-s1"), ("lc1-b174-kda", "lc1-b174-kda-s1")),
}
KEYS = ("macro16", "macro15_noptb", "macro14_noptb_nowt", "uncheatable", "ptb", "wikitext_103")

if __name__ == "__main__":
    runs = pairs.load(sys.argv[1])
    out = {}
    for label, ((a0, a1), (b0, b1)) in CALLS.items():
        A0, A1, B0, B1 = (snr.values(runs[n]) for n in (a0, a1, b0, b1))
        out[label] = {"runs": [a0, a1, b0, b1],
                      "delta": {k: {"seed0": A0[k] - B0[k], "seed1": A1[k] - B1[k], "mean": (A0[k] - B0[k] + A1[k] - B1[k]) / 2} for k in KEYS}}
        print("==", label)
        for k in KEYS:
            d = out[label]["delta"][k]
            print(f"   {k:20s} {d['seed0']:+.4f} / {d['seed1']:+.4f}  mean {d['mean']:+.4f}")
    json.dump(out, open(sys.argv[2], "w"), indent=1)
