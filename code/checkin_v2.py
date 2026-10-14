#!/usr/bin/env python3
"""70% 检查的替代(2026-10-14):hero 在第 257,553 步切到 16K,之后没有 4K 评测,所以在切换时补做预注册第 2 节承诺的三项:
  1. 冻结后每次新 4K 评测在冻结轨迹分布下的 PIT;
  2. 落在 80% 区间里的次数;
  3. 用新数据重估的残差斜率 b(和冻结时的 b 对比)。
做法:冻结的 v2 代码和冻结快照原样使用;轨迹分布用冻结时的参数(κ_max、ε、趋势、评测噪声、PTB 模型,取自
results/v2/forecast_summary_v2.json)和同一个随机种子重新抽样,只是把输出点换成新评测所在的 p。
新评测直接从 W&B 读(匿名 GraphQL)。不改动冻结的预测。

用法(在仓库的 code/ 目录下):
  python checkin_v2.py ../data/snapshot_2026-10-09T0440Z ../addenda
"""

import csv
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

import forecast_step3 as F
import forecast_v2 as V
from reproduce import load

ROOT = os.environ.get("FORECAST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EP = "https://api.wandb.ai/graphql"
RUN = "hero-mfu30-step231k"
LAST_4K_EVAL = 254_999


def fetch_new_evals(lo, hi):
    def gql(query, variables):
        body = json.dumps({"query": query, "variables": variables}).encode()
        req = urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())["data"]["project"]["run"]
    keys = sorted(k for k in gql("""query($n: String!) { project(name: "marin_moe", entityName: "marin-community") {
        run(name: $n) { historyKeys } } }""", {"n": RUN})["historyKeys"]["keys"]
                  if re.match(r"^eval_dropless/paloma/[^/]+-llama3/loss$", k)) + ["eval_dropless/paloma/macro_loss"]
    spec = json.dumps({"keys": ["_step", *keys], "samples": 100000, "minStep": lo, "maxStep": hi})
    rows = gql("""query($n: String!, $specs: [JSONString!]!) { project(name: "marin_moe", entityName: "marin-community") {
        run(name: $n) { sampledHistory(specs: $specs) } } }""", {"n": RUN, "specs": [spec]})["sampledHistory"][0]
    return pd.DataFrame([{"category": "hero", "run": RUN, "step": int(r["_step"]), "key": k, "value": float(r[k])}
                         for r in rows for k in keys])


def weighted_ab(rows):
    w = np.array([r["w"] for r in rows])
    w = w / w.sum()
    return {k: float(np.dot(w, [r[k] for r in rows])) for k in ("a", "b", "se_a", "se_b")}


def main(snap, outdir):
    summ = json.load(open(os.path.join(ROOT, "results", "v2", "forecast_summary_v2.json")))
    long, runs = load(snap)
    long = V.dedupe_latest(long)
    lads = F.build_ladders(long, runs)

    # 冻结时的锚定(数据截至 227,999)
    hp15, hy15 = F.hero_series(long, "macro15")
    p_now = float(hp15[-1])
    assert abs(p_now - 228_000 / F.HERO_STEPS) < 1e-12, p_now

    # 新评测
    new = fetch_new_evals(int(round(p_now * F.HERO_STEPS)), LAST_4K_EVAL)
    new_steps = sorted(new.step.unique())
    long_new = pd.concat([long, new], ignore_index=True)
    hp16n, hy16n = F.hero_series(long_new, "macro16")
    y16 = {int(round(p * F.HERO_STEPS)) - 1: y for p, y in zip(hp16n, hy16n)}
    p_new = [(s + 1) / F.HERO_STEPS for s in new_steps]

    # 1) 冻结分布下的 PIT:冻结参数 + 同一种子,输出点换成新评测的 p;另取 1% 网格点核对是否复现冻结的轨迹文件
    me, pt = summ["method_error"]["macro15_incl_67b"], summ["ptb_model"]
    grid = [0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66]
    p_out = sorted(set(p_new) | set(grid))
    rows15 = V.hero_rows(lads, "macro15", hp15, hy15, p_now, p_out)
    rng = np.random.default_rng(V.SEED)
    _, tr15 = V.mc_v2(rows15, p_now, summ["drift"]["kappa_max"], me["sigma"], me["nu"], summ["eval_noise"]["macro15"],
                      V.N_MC, rng, p_out, trend=me["trend"])
    _, d_tr = V.sample_D_v2(pt, V.N_MC, p_now, p_out, rng)
    tr16 = {p: tr15[p] + d_tr[p] / 16 for p in p_out}
    frozen = {round(float(r["fraction"]), 2): r for r in csv.DictReader(open(os.path.join(ROOT, "results", "v2", "trajectory_official_macro16_v2.csv")))}
    repro = {}
    for p in grid:
        q = F.quantiles(tr16[p])
        repro[p] = {c: q[c] - float(frozen[p][c]) for c in ("q025", "q10", "median", "q90", "q975")}
    max_dev = max(abs(v) for d in repro.values() for v in d.values())

    pits = []
    for s, p in zip(new_steps, p_new):
        x = tr16[p]
        y = y16[s]
        q = F.quantiles(x)
        pits.append({"step": int(s), "p": p, "macro16": y, "pit": float(np.mean(x <= y)),
                     "in80": bool(q["q10"] <= y <= q["q90"]), "in95": bool(q["q025"] <= y <= q["q975"])})

    # 3) 用新数据重估 a、b(15 子集目标,和冻结时同一套行)
    hp15n, hy15n = F.hero_series(long_new, "macro15")
    p_now_n = float(hp15n[-1])
    rows_f = V.hero_rows(lads, "macro15", hp15, hy15, p_now, [])
    rows_n = V.hero_rows(lads, "macro15", hp15n, hy15n, p_now_n, [])
    ab_frozen, ab_new = weighted_ab(rows_f), weighted_ab(rows_n)
    w = np.array([r["w"] for r in rows_n]) / sum(r["w"] for r in rows_n)
    bf, bn = np.array([r["b"] for r in rows_f]), np.array([r["b"] for r in rows_n])
    db = bn - bf
    # 两次 WikiText 大跌(239,999 和 254,999)去掉以后的 b
    drop = np.isin(np.round(hp15n * F.HERO_STEPS).astype(int) - 1, [239_999, 254_999])
    ab_no_wt = weighted_ab(V.hero_rows(lads, "macro15", hp15n[~drop], hy15n[~drop], p_now_n, []))
    # 漂移项 κ·b·(1 − p_now) 对终点中心的影响:新拟合减冻结拟合,κ 取先验均值和上限
    kmax = summ["drift"]["kappa_max"]
    shift = {f"kappa={k:g}": k * (ab_new["b"] * (1 - p_now_n) - ab_frozen["b"] * (1 - p_now)) for k in (kmax / 2, kmax)}
    b_rows = {"n_rows": len(rows_n), "b_sd_across_rows_frozen": float(np.sqrt(np.cov(bf, aweights=w))),
              "b_sd_across_rows_new": float(np.sqrt(np.cov(bn, aweights=w))),
              "change_min": float(db.min()), "change_max": float(db.max()), "change_weighted_mean": float(np.dot(w, db)),
              "rows_where_b_fell": int((db < 0).sum()), "b_without_wikitext_drop_evals": ab_no_wt["b"],
              "end_center_shift_from_drift": shift, "note_se": "se_b is the row-weighted mean of each row's OLS SE (within-row only)"}

    out = {"frozen_p_now": p_now, "new_evals": [int(s) for s in new_steps], "pit": pits,
           "in80": sum(r["in80"] for r in pits), "in95": sum(r["in95"] for r in pits), "n": len(pits),
           "trajectory_reproduction_max_abs_dev": max_dev, "trajectory_reproduction": repro,
           "residual_fit_frozen": {**ab_frozen, "p_now": p_now, "n_evals": int(((hp15 >= F.POST_SWAP_P) & (hp15 <= p_now + 1e-9)).sum())},
           "residual_fit_new": {**ab_new, "p_now": p_now_n, "n_evals": int(((hp15n >= F.POST_SWAP_P) & (hp15n <= p_now_n + 1e-9)).sum())},
           "b_details": b_rows}
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "2026-10-14_checkin.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"reproduction of the frozen trajectory file (p = 0.59..0.66): max |dev| = {max_dev:.5f}")
    for r in pits:
        print(f"  {r['step']}: macro16 {r['macro16']:.5f}  PIT {r['pit']:.4f}  in80 {r['in80']}  in95 {r['in95']}")
    print(f"in 80%: {out['in80']}/{out['n']}, in 95%: {out['in95']}/{out['n']}")
    for k in ("residual_fit_frozen", "residual_fit_new"):
        d = out[k]
        print(f"{k}: a {d['a']:+.5f} (SE {d['se_a']:.5f}), b {d['b']:+.5f} (SE {d['se_b']:.5f}), p_now {d['p_now']:.4f}, n {d['n_evals']}")
    print(json.dumps(b_rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
