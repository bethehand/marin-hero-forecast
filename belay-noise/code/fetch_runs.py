"""Fetch every finished Belay run (display name ^lc1-) from the public W&B project marin-community/marin_moe.

Keeps, per run: name, createdAt, the flattened config without per-run bookkeeping keys, and the summary
metrics whose names start with "eval" (final evals) plus _step and _runtime. Writes JSON lines.

usage: python3 -I fetch_runs.py <out.jsonl>
"""
import os, json, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb

VOLATILE = ("_wandb.", "trainer.value.trainer.checkpointer.", "trainer.value.trainer.tracker.")
VOLATILE_EXACT = {"trainer.value.trainer.id", "trainer.value.trainer.load_checkpoint_path"}


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flatten(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def main(out_path):
    q = """query($f: JSONString, $after: String, $first: Int) { project(name: "marin_moe", entityName: "marin-community") {
      runs(filters: $f, after: $after, first: $first, order: "-created_at") {
        pageInfo { endCursor hasNextPage } edges { node { name state createdAt config summaryMetrics } } } } }"""
    f = json.dumps({"$and": [{"display_name": {"$regex": "^lc1-"}}, {"state": "finished"}]})
    after, n = None, 0
    with open(out_path, "w") as fo:
        while True:
            r = wb.gql(q, {"f": f, "after": after, "first": 40})["project"]["runs"]
            for e in r["edges"]:
                node = e["node"]
                cfg = flatten(json.loads(node["config"] or "{}"))
                cfg = {k: v for k, v in cfg.items() if not k.startswith(VOLATILE) and k not in VOLATILE_EXACT}
                s = json.loads(node["summaryMetrics"] or "{}")
                ev = {k: v for k, v in s.items() if k.startswith("eval") or k in ("_step", "_runtime")}
                fo.write(json.dumps({"name": node["name"], "state": node["state"], "createdAt": node["createdAt"],
                                     "cfg": cfg, "ev": ev}) + "\n")
                n += 1
            print(f"{n} runs", flush=True)
            if not r["pageInfo"]["hasNextPage"]:
                break
            after = r["pageInfo"]["endCursor"]
            time.sleep(0.5)
    print("fetched at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "->", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
