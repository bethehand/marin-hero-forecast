"""Pull per-step train loss for the #9615 same-seed reruns (ctx9615-{ctl,tip}-*-s0-01), read-only."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb
from pull_loss import history_range, KEYS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RUNS = [f"ctx9615-{v}-{c}-s0-01" for v in ("ctl", "tip") for c in ("4k", "8k", "16k")]


def main():
    found = {r["displayName"]: r["name"] for r in wb.list_runs(r"^ctx9615-")}
    print("ctx9615 runs on W&B:", sorted(found))
    res = {}
    for disp in RUNS:
        name = found.get(disp)
        if not name:
            print("missing", disp)
            continue
        rows = history_range(name, KEYS, 179990, 180210)
        res[disp] = {str(r["_step"]): {k: r.get(k) for k in KEYS} for r in rows}
        print(disp, "rows:", len(rows))
    with open(os.path.join(OUT, "loss_reruns_180k.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
