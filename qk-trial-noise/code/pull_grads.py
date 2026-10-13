"""Pull per-parameter grad norms (grad/norm/*) for the 16K runs restored from step 180k: three seeds and two same-seed reruns (read-only)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb
from pull_loss import history_range

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RUNS = ["mhep-ctx16k-s0-20260930", "mhep-ctx16k-s1-20260930", "mhep-ctx16k-s2-20260930",
        "ctx9615-ctl-16k-s0-01", "ctx9615-tip-16k-s0-01"]


def main():
    names = {r["displayName"]: r["name"] for r in wb.list_runs(r"^(mhep-ctx16k|ctx9615-(ctl|tip)-16k)")}
    meta = wb.run_meta(names[RUNS[0]])
    keys = sorted(k for k in (meta["historyKeys"].get("keys") or {}) if k.startswith("grad/norm/"))
    print("grad keys:", len(keys))
    res = {}
    for disp in RUNS:
        rows = {}
        for i in range(0, len(keys), 120):
            chunk = keys[i:i + 120]
            for r in history_range(names[disp], chunk, 180000, 180099):
                rows.setdefault(str(r["_step"]), {}).update({k: r.get(k) for k in chunk if r.get(k) is not None})
        res[disp] = rows
        print(disp, "steps:", len(rows), "keys at first step:", len(rows.get("180000", {})))
    with open(os.path.join(OUT, "grads_16k_180k.json"), "w") as f:
        json.dump({"keys": keys, "runs": res}, f)


if __name__ == "__main__":
    main()
