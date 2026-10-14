#!/usr/bin/env python3
"""计分(2026-10-14):hero 在第 257,553 步切到 16K(marin-community/marin#10101),按 preregistration_v2.md 第 2 节条件 2,
用切换前最后一次 4K 评测给冻结的 v2 计分,同时报告 v1 和团队 8/19 轨迹的得分,以及冻结后每次 4K 评测落在 v2 区间的哪里。
不改动冻结的预测。只读 W&B 公开数据(匿名 GraphQL)和仓库里冻结的轨迹文件。

用法:
  python score_switch.py ../addenda
"""

import csv
import json
import os
import re
import sys
import urllib.request

ROOT = os.environ.get("FORECAST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EP = "https://api.wandb.ai/graphql"
RUN = "hero-mfu30-step231k"          # 4K 父 run,停在 step-257553
TEAM_RUN = "hero-forecast-seqlen4096-fc"  # 团队 8/19 预注册的逐 5% 轨迹
N = 390251
FREEZE_LAST_STEP = 227999            # v2 冻结时数据截止的评测
TAUS = (0.025, 0.10, 0.50, 0.90, 0.975)
Q = ("q025", "q10", "median", "q90", "q975")
MACRO = "eval_dropless/paloma/macro_loss"


def gql(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]["project"]["run"]


def history(run, keys, lo=None, hi=None):
    spec = {"keys": ["_step", *keys], "samples": 100000}
    if lo is not None:
        spec.update(minStep=lo, maxStep=hi)
    q = """query($n: String!, $specs: [JSONString!]!) { project(name: "marin_moe", entityName: "marin-community") {
      run(name: $n) { sampledHistory(specs: $specs) } } }"""
    return sorted(gql(q, {"n": run, "specs": [json.dumps(spec)]})["sampledHistory"][0], key=lambda r: r["_step"])


def trajectory(path):
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(os.path.join(ROOT, path)))]


def at(rows, p, cols):
    """在 p 处对轨迹线性插值。"""
    for a, b in zip(rows, rows[1:]):
        if a["fraction"] <= p <= b["fraction"]:
            w = (p - a["fraction"]) / (b["fraction"] - a["fraction"])
            return {c: a[c] + w * (b[c] - a[c]) for c in cols}
    raise ValueError(f"p={p} outside the trajectory")


def quantile_score(y, q):
    """2 × 五个冻结分位数上的平均 pinball loss;点预测时等于绝对误差。"""
    pin = [(y - q[c]) * (t - (1.0 if y < q[c] else 0.0)) for t, c in zip(TAUS, Q)]
    return 2 * sum(pin) / len(pin)


def band(y, q):
    if y < q["q025"]:
        return "below 2.5%"
    if y < q["q10"]:
        return "2.5-10%"
    if y < q["median"]:
        return "10-50%"
    if y <= q["q90"]:
        return "50-90%"
    return "90-97.5%" if y <= q["q975"] else "above 97.5%"


def main(outdir):
    keys_q = """query($n: String!) { project(name: "marin_moe", entityName: "marin-community") { run(name: $n) { historyKeys } } }"""
    sub_keys = sorted(k for k in gql(keys_q, {"n": RUN})["historyKeys"]["keys"]
                      if re.match(r"^eval_dropless/paloma/.+-llama3/loss$", k))
    assert len(sub_keys) == 16, len(sub_keys)
    evals = []
    for r in history(RUN, [MACRO, *sub_keys], FREEZE_LAST_STEP + 1, 257553):
        subs = {k.split("/")[2][: -len("-llama3")]: r[k] for k in sub_keys}
        assert abs(sum(subs.values()) / 16 - r[MACRO]) < 2e-6
        evals.append({"step": r["_step"], "macro16": r[MACRO],
                      "companion15": sum(v for k, v in subs.items() if k != "ptb") / 15,
                      "macro14": sum(v for k, v in subs.items() if k not in ("ptb", "wikitext_103")) / 14,
                      "ptb": subs["ptb"], "wikitext_103": subs["wikitext_103"]})
    last = evals[-1]
    p = (last["step"] + 1) / N

    team_rows = history(TEAM_RUN, [MACRO, "forecast/training_pct"])
    team_traj = [{"fraction": r["forecast/training_pct"] / 100, "value": r[MACRO]} for r in team_rows]
    team = at(team_traj, p, ("value",))["value"]

    files = {"v2": ("results/v2/trajectory_official_macro16_v2.csv", "results/v2/trajectory_macro15_v2.csv"),
             "v1": ("results/step5/trajectory_official_macro16.csv", "results/step5/trajectory_macro15.csv")}
    score = {"scored_eval_step": last["step"], "p": p, "realized_macro16": last["macro16"],
             "realized_companion15": last["companion15"], "forecasts": {}}
    for tag, (f16, f15) in files.items():
        for target, f, y in (("primary", f16, last["macro16"]), ("companion", f15, last["companion15"])):
            q = at(trajectory(f), p, Q)
            score["forecasts"][f"{tag} {target}"] = {**q, "in80": q["q10"] <= y <= q["q90"],
                                                     "in95": q["q025"] <= y <= q["q975"], "band": band(y, q),
                                                     "quantile_score": quantile_score(y, q)}
    score["forecasts"]["team Aug 19 primary"] = {"value": team, "quantile_score": abs(last["macro16"] - team)}

    v2 = trajectory(files["v2"][0])
    since = []
    for e in evals:
        q = at(v2, (e["step"] + 1) / N, Q)
        since.append({**e, "p": (e["step"] + 1) / N, "v2_median": q["median"], "v2_band": band(e["macro16"], q),
                      "ptb_plus_wikitext": e["ptb"] + e["wikitext_103"]})
    prev = evals[-2]
    out = {"score": score, "evals_since_freeze": since,
           "in80": sum(s["v2_band"] in ("10-50%", "50-90%") for s in since), "n": len(since),
           "last_eval_drop": {"macro16": last["macro16"] - prev["macro16"],
                              "wikitext_103": last["wikitext_103"] - prev["wikitext_103"],
                              "wikitext_share_of_macro16": (last["wikitext_103"] - prev["wikitext_103"]) / 16}}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "2026-10-14_switch_score.json"), "w") as f:
        json.dump(out, f, indent=1)

    print(f"scored eval: step {last['step']} (p = {p:.6f}); realized macro16 {last['macro16']:.5f}, companion {last['companion15']:.5f}")
    for k, v in score["forecasts"].items():
        print(f"  {k}: " + ", ".join(f"{c} {v[c]:.5f}" for c in (*Q, "value", "quantile_score") if c in v)
              + (f", band {v['band']}" if "band" in v else ""))
    print("evals since the freeze (v2 primary):")
    for s in since:
        print(f"  {s['step']}: {s['macro16']:.5f} vs median {s['v2_median']:.5f} -> {s['v2_band']}; "
              f"ptb+wikitext {s['ptb_plus_wikitext']:.4f}; macro14 {s['macro14']:.4f}")
    print(f"in the 80% band: {out['in80']}/{out['n']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "addenda"))
