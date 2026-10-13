"""Pull per-step train loss for the Sept 30 mhep-ctx runs and the hero over the same steps (read-only W&B)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RUNS = [f"mhep-ctx{c}-s{s}-20260930" for c in ("4k", "8k", "16k") for s in (0, 1, 2)]
HERO = "hero-fa4sm100-nomask-step146k"
KEYS = ["train/loss", "train/cross_entropy_loss"]


def history_range(name, keys, lo, hi):
    q = """query($n: String!, $specs: [JSONString!]!) { project(name: "marin_moe", entityName: "marin-community") {
      run(name: $n) { sampledHistory(specs: $specs) } } }"""
    spec = json.dumps({"keys": ["_step", *keys], "samples": 100000, "minStep": lo, "maxStep": hi})
    rows = wb.gql(q, {"n": name, "specs": [spec]})["project"]["run"]["sampledHistory"][0]
    return sorted(rows, key=lambda r: r["_step"])


def main():
    res = {}
    for name in RUNS + [HERO]:
        rows = history_range(name, KEYS, 179990, 180210)
        res[name] = {str(r["_step"]): {k: r.get(k) for k in KEYS} for r in rows}
        st = sorted(int(s) for s in res[name])
        print(name, "rows:", len(st), "steps:", (st[0], st[-1]) if st else None)
    with open(os.path.join(OUT, "loss_180k.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
