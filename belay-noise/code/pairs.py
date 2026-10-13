"""Seed pairs from the Belay runs: <name> (seed 0) and <name>-s1 (seed 1), identical config apart from the seed,
both finished at num_train_steps with all 16 Paloma subsets and the Uncheatable subsets logged."""
import json, re

SEED = "trainer.value.trainer.seed"
NS = "trainer.value.trainer.num_train_steps"
CLEAN_EXCLUDE = ("ptb", "wikitext_103")


def subsets(ev, family):
    pre = f"eval/{family}/"
    return {k[len(pre):].split("/")[0].replace("-hero-bpe-v16384", ""): v for k, v in ev.items()
            if k.startswith(pre) and k.endswith("/loss") and k.count("/") == 3}


def load(path):
    runs = {}
    for line in open(path):
        r = json.loads(line)
        ev = r["ev"]
        ns = r["cfg"].get(NS)
        # the final eval is logged at step num_train_steps (older runs) or num_train_steps - 1 (newer runs)
        if "eval/paloma/macro_loss" not in ev or ns is None or ev.get("_step") not in (ns, ns - 1):
            continue
        # drop per-run bookkeeping (output paths etc.) whose value names the run itself
        r["cfg"] = {k: v for k, v in r["cfg"].items() if r["name"] not in str(v)}
        r["pal"], r["unc"] = subsets(ev, "paloma"), subsets(ev, "uncheatable_eval")
        if len(r["pal"]) == 16:
            runs[r["name"]] = r
    return runs


def seed_pairs(runs):
    out = []
    for name, r1 in runs.items():
        if r1["cfg"].get(SEED) != 1:
            continue
        # the seed-1 run is named after its seed-0 partner with an "-s1" token, at the end or in the middle
        cands = {re.sub(r"-s1(?=-|$)", "", name, count=1), re.sub(r"-s1(?=-|$)", "-s0", name, count=1)} - {name}
        r0 = next((runs[c] for c in sorted(cands) if c in runs and runs[c]["cfg"].get(SEED) == 0), None)
        if r0 is None:
            continue
        c0 = {k: v for k, v in r0["cfg"].items() if k != SEED}
        c1 = {k: v for k, v in r1["cfg"].items() if k != SEED}
        if c0 != c1 or set(r0["unc"]) != set(r1["unc"]):
            continue
        m = re.match(r"lc1-b(\d+)-", r0["name"])
        # runs without a batch number are grouped by launch date
        out.append({"name": r0["name"], "batch": int(m.group(1)) if m else "day-" + r0["createdAt"][:10],
                    "hidden": r0["cfg"].get("model.value.hidden_dim"), "steps": r0["cfg"].get(NS),
                    "created": r0["createdAt"], "r0": r0, "r1": r1})
    return out


def config_pairs(runs, exclude=()):
    """Looser pairing for robustness: each seed-1 run with the seed-0 run of identical config created closest in time."""
    def key(r):
        return json.dumps({k: v for k, v in r["cfg"].items() if k != SEED}, sort_keys=True, default=str)
    by_key = {}
    for r in runs.values():
        if r["cfg"].get(SEED) == 0:
            by_key.setdefault(key(r), []).append(r)
    out = []
    for r1 in runs.values():
        if r1["cfg"].get(SEED) != 1 or r1["name"] in exclude:
            continue
        cands = [r for r in by_key.get(key(r1), []) if set(r["unc"]) == set(r1["unc"])]
        if not cands:
            continue
        r0 = min(cands, key=lambda r: abs(_t(r["createdAt"]) - _t(r1["createdAt"])))
        m = re.match(r"lc1-b(\d+)-", r1["name"])
        out.append({"name": r0["name"], "batch": int(m.group(1)) if m else "day-" + r1["createdAt"][:10],
                    "hidden": r0["cfg"].get("model.value.hidden_dim"), "steps": r0["cfg"].get(NS),
                    "created": r1["createdAt"], "r0": r0, "r1": r1})
    return out


def _t(s):
    import datetime as dt
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").timestamp()
