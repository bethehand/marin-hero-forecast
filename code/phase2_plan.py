#!/usr/bin/env python3
"""第二阶段(从第 312,192 步起换成冷却数据配比;学习率从预热结束起一直线性下降,在这一步不变)开始之前,
把终点分析要比的预测值固定下来。

算的是"从第 311,999 步(第二阶段前最后一次评测)到训练结束,损失还会降多少":
  1. 阶梯定律(v2 的逐 5% 拟合,只用定律,不加 hero 自己的漂移项):
     有下限(E 取 v2 的网格)和无下限(E = 0)两种;平滑后(v2 的做法)和不平滑(直接取 80%、100% 两点的拟合)两种。
     目标:M14(去掉 PTB 和 WikiText-103)、M15(去掉 PTB)、M16(官方 16 子集)。
  2. v2 冻结轨迹(2026-10-09)的中位数路径。
  3. 团队 8/19 的预注册轨迹(W&B run hero-forecast-seqlen4096-fc)。
另外记录:
  4. 切换时 4K→16K 的差值:qk16k-a1/a0 两个试验(从第 252,000 步恢复,16K)在第 252,049、252,099 步,
     减去 hero 第 251,999 步的 4K 评测;分 16 个子集和 M14/M15/M16。
  5. hero 16K 段(第 257,999 步起)各次评测的 M14/M15/M16/Uncheatable,和去趋势后的噪声。

阶梯数据用冻结时的快照(阶梯早已跑完,没有新数据);4 和 5 从 W&B 现拉;5 的原始数据存进输出目录,4 只存算好的差值。
噪声和判读区间只用写计划时已有的 16K 评测(到第 305,999 步),之后新增的评测不改变它们。
用法:
  python phase2_plan.py ../data/snapshot_2026-10-09T0440Z ../results/phase2
"""

import csv
import json
import os
import re
import sys
import urllib.request

import numpy as np

import forecast_step3 as F
import forecast_v2 as V
from reproduce import load

HERE = os.path.dirname(os.path.abspath(__file__))
HERO_STEPS = 390_251
START_STEP = 311_999  # 第二阶段(第 312,192 步)之前的最后一次评测
NOISE_LAST = 305_999  # 噪声和判读区间只用写计划时已有的 16K 评测
P_C = (START_STEP + 1) / HERO_STEPS
WANDB = "https://api.wandb.ai/graphql"
PROJECT = {"name": "marin_moe", "entityName": "marin-community"}
HERO_16K_RUN = "hero-ctx16k-step257k"
HERO_4K_RUN = "hero-mfu30-step231k"
TEAM_RUN = "hero-forecast-seqlen4096-fc"
ARMS = {"a1 (qk_mult 1.48, = hero after the switch)": "qk16k-a1-step252k", "a0 (qk_mult 1.3, = hero before)": "qk16k-a0-step252k"}
ARM_STEPS = (252_049, 252_099)
SUBSET_RE = r"^eval_dropless/paloma/[^/]+-llama3/loss$"
PTB, WT = "eval_dropless/paloma/ptb-llama3/loss", "eval_dropless/paloma/wikitext_103-llama3/loss"
MACRO, UNCH = "eval_dropless/paloma/macro_loss", "eval_dropless/uncheatable_eval/macro_loss"


# ---------------- W&B ----------------
def gql(query, variables):
    req = urllib.request.Request(WANDB, data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())["data"]


def summary(run):
    q = "query($p:String!,$e:String!,$n:String!){project(name:$p,entityName:$e){run(name:$n){summaryMetrics}}}"
    return json.loads(gql(q, {"p": PROJECT["name"], "e": PROJECT["entityName"], "n": run})["project"]["run"]["summaryMetrics"])


def history(run, keys, lo, hi):
    q = ("query($p:String!,$e:String!,$n:String!,$specs:[JSONString!]!)"
         "{project(name:$p,entityName:$e){run(name:$n){sampledHistory(specs:$specs)}}}")
    out = {}
    for k in keys:
        spec = json.dumps({"keys": ["_step", k], "samples": 5000, "minStep": lo, "maxStep": hi})
        rows = gql(q, {"p": PROJECT["name"], "e": PROJECT["entityName"], "n": run, "specs": [spec]})["project"]["run"]["sampledHistory"][0]
        for r in rows:
            if r.get(k) is not None:
                out.setdefault(int(r["_step"]), {})[k] = float(r[k])
    return out


def macros(v, subs):
    """一次评测的 16 个子集 → M16、M15(去掉 PTB)、M14(去掉 PTB 和 WikiText-103)。"""
    return {"M16": float(np.mean([v[k] for k in subs])),
            "M15": float(np.mean([v[k] for k in subs if k != PTB])),
            "M14": float(np.mean([v[k] for k in subs if k not in (PTB, WT)]))}


# ---------------- 阶梯定律给的降幅 ----------------
def metric_steps14(long, run):
    d = long[long.run == run]
    sub = d[d.key.str.match(SUBSET_RE) & ~d.key.isin([PTB, WT])]
    w = sub.pivot_table(index="step", columns="key", values="value").dropna()
    assert w.shape[1] == 14, (run, w.shape)
    return sorted(zip(w.index.astype(int), w.mean(axis=1).astype(float)))


def ladder_drops(lads, target, floor, smooth):
    """逐 5% 拟合 L = E + A·(C·p)^s,在 hero 算力处取值,返回每个集成成员的 f(p_c) − f(1) 和权重(权重同 v2)。"""
    spec = {k: (w, list(lads[k]["rungs"])) for k, w in F.LADDER_W.items()}
    profs = V.d2048_profiles(lads, target)
    rows = []
    for lad_name, (lw, widths) in spec.items():
        lad = lads[lad_name]
        variants = list(profs.items()) if "d2048" in widths else [(None, None)]
        for conv, cw in F.CONV_W.items():
            Es = F.e_grid(conv, step=0.01) if floor else np.array([0.0])
            for _, prof in variants:
                grids = dict(lad["grids"][target])
                if prof is not None:
                    grids["d2048"] = V.extend_d2048_profile(grids["d2048"], prof)
                for E in Es:
                    t = F.traj(grids, lad["rungs"], widths, conv, E, F.HERO[conv])
                    if smooth:
                        f = F.smooth(t)
                        d, lv = float(f(P_C) - f(1.0)), float(f(1.0))
                    else:  # 不平滑:80% 和 100% 两个拟合点之差(p_c = 0.7995)
                        d, lv = float(t[0.8] - t[1.0]), float(t[1.0])
                    rows.append((d, lw * cw / (len(Es) * len(variants)), lad_name, lv))
    d = np.array([r[0] for r in rows])
    w = np.array([r[1] for r in rows])
    w = w / w.sum()
    order = np.argsort(d)
    cw = np.cumsum(w[order])
    q = lambda x: float(d[order][min(np.searchsorted(cw, x), len(d) - 1)])
    by = {lad: float(w[[r[2] == lad for r in rows]] @ d[[r[2] == lad for r in rows]]
                     / w[[r[2] == lad for r in rows]].sum()) for lad in F.LADDER_W}
    return {"mean": float(w @ d), "q05": q(0.05), "q50": q(0.5), "q95": q(0.95), "by_ladder": by, "members": len(rows),
            "end_level_unanchored": float(w @ np.array([r[3] for r in rows]))}


def traj_at(rows, p):
    for (p0, v0), (p1, v1) in zip(rows, rows[1:]):
        if p0 - 1e-12 <= p <= p1 + 1e-12:
            return v0 + (p - p0) / (p1 - p0) * (v1 - v0)
    raise ValueError(p)


def main(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    res = {"start_step": START_STEP, "p_c": P_C}

    # 1. 阶梯定律
    long, runs = load(snap)
    long = V.dedupe_latest(long)
    lads = F.build_ladders(long, runs)
    for lad in lads.values():
        lad["grids"]["macro14"] = {w: F.snap(metric_steps14(long, v["run"]), int(runs.loc[v["run"], "stop_after_steps"]), lad["rule"])
                                   for w, v in lad["rungs"].items()}
    names = {"macro14": "M14", "macro15": "M15", "macro16": "M16"}
    res["ladder"] = {}
    for target, m in names.items():
        for floor in (True, False):
            for smooth in (True, False):
                key = f"{m}/{'floor' if floor else 'no floor'}/{'smoothed' if smooth else 'raw 80-100%'}"
                res["ladder"][key] = ladder_drops(lads, target, floor, smooth)

    # 2. v2 冻结轨迹
    res["v2"] = {}
    for m, f in (("M16", "trajectory_official_macro16_v2.csv"), ("M15", "trajectory_macro15_v2.csv")):
        rows = [(float(r["fraction"]), float(r["median"])) for r in csv.DictReader(open(os.path.join(HERE, "..", "results", "v2", f)))]
        a, b = traj_at(rows, P_C), traj_at(rows, 1.0)
        res["v2"][m] = {"at_start": a, "at_end": b, "drop": a - b}

    # 3. 团队 8/19 轨迹
    team = history(TEAM_RUN, [MACRO, "forecast/training_pct"], 0, 10 ** 9)
    trows = sorted((v["forecast/training_pct"] / 100, v[MACRO]) for v in team.values() if MACRO in v and "forecast/training_pct" in v)
    a, b = traj_at(trows, P_C), traj_at(trows, 1.0)
    res["team_aug19"] = {"M16": {"at_start": a, "at_end": b, "drop": a - b}}
    json.dump(trows, open(os.path.join(outdir, "team_aug19_trajectory.json"), "w"), indent=1)

    # 4. 切换时 4K→16K 的差值
    subs = sorted(k for k in summary(HERO_16K_RUN) if re.match(SUBSET_RE, k))
    assert len(subs) == 16, subs
    h4 = history(HERO_4K_RUN, subs + [MACRO], 251_999, 251_999)[251_999]
    off = {"hero_4k_251999": {**macros(h4, subs), "logged": h4[MACRO]}, "arms": {}}
    for label, run in ARMS.items():
        a = history(run, subs + [MACRO], 252_000, 252_200)
        vals = [macros(a[s], subs) for s in ARM_STEPS]
        off["arms"][label] = {"run": run, "steps": ARM_STEPS,
                              "offset": {m: float(np.mean([v[m] for v in vals])) - off["hero_4k_251999"][m] for m in ("M16", "M15", "M14")},
                              "offset_logged_macro": float(np.mean([a[s][MACRO] for s in ARM_STEPS])) - h4[MACRO],
                              "per_subset": {k.split("/")[2]: float(np.mean([a[s][k] for s in ARM_STEPS])) - h4[k] for k in subs}}
    res["switch_offset"] = off

    # 5. hero 16K 段:各指标和噪声
    ev = history(HERO_16K_RUN, subs + [MACRO, UNCH], 257_000, 10 ** 9)
    steps = sorted(s for s in ev if s <= NOISE_LAST and all(k in ev[s] for k in subs + [UNCH]))
    table = [{"step": s, **macros(ev[s], subs), "M16_logged": ev[s][MACRO], "MU": ev[s][UNCH], "PTB": ev[s][PTB], "WT": ev[s][WT]}
             for s in steps]
    json.dump(table, open(os.path.join(outdir, "hero_16k_evals.json"), "w"), indent=1)
    p = np.array([(s + 1) / HERO_STEPS for s in steps])
    res["noise_16k"] = {"evals": [steps[0], steps[-1], len(steps)]}
    for m in ("M14", "M15", "M16", "MU"):
        y = np.array([r[m] for r in table])
        res["noise_16k"][m] = {f"deg{d}": float(np.std(y - np.polyval(np.polyfit(p, y, d), p), ddof=d + 1)) for d in (1, 2)}
    res["latest_eval"] = table[-1]

    # 判读区间:四种阶梯算法 5–95% 范围的并集,两侧各放宽 2 倍 Δ 的标准差。
    # S 取 5 次评测直线在末点的值(标准误 = σ·sqrt(1/5 + 4/10) = 0.775σ),Y 取 1 次评测,所以 sd(Δ) = σ·sqrt(1.6)。
    res["bands"] = {}
    for m in ("M14", "M15", "M16"):
        sig = max(res["noise_16k"][m].values())
        sd = sig * np.sqrt(1.6)
        rows = [v for k, v in res["ladder"].items() if k.startswith(m + "/")]
        lo, hi = min(v["q05"] for v in rows), max(v["q95"] for v in rows)
        res["bands"][m] = {"sigma_eval": sig, "sd_drop": float(sd), "ladder_union": [lo, hi], "band": [lo - 2 * sd, hi + 2 * sd]}

    json.dump(res, open(os.path.join(outdir, "plan_numbers.json"), "w"), indent=1)

    # 摘要
    print(f"p_c = {P_C:.5f} (step {START_STEP})")
    for k, v in res["ladder"].items():
        print(f"ladder {k:32s} drop {v['mean']:.4f}  5–95% [{v['q05']:.4f}, {v['q95']:.4f}]  "
              + ", ".join(f"{a} {b:.4f}" for a, b in v["by_ladder"].items()))
    for m, v in res["v2"].items():
        print(f"v2 frozen {m}: {v['at_start']:.4f} → {v['at_end']:.4f}, drop {v['drop']:.4f}")
    t = res["team_aug19"]["M16"]
    print(f"team Aug 19 M16: {t['at_start']:.4f} → {t['at_end']:.4f}, drop {t['drop']:.4f}")
    for label, a in off["arms"].items():
        print(f"switch offset {label}: " + ", ".join(f"{m} {v:+.4f}" for m, v in a["offset"].items())
              + f", logged macro {a['offset_logged_macro']:+.4f}")
    print("noise (detrended SD, 16K evals " + f"{steps[0]}–{steps[-1]}, n={len(steps)}): "
          + ", ".join(f"{m} {res['noise_16k'][m]['deg1']:.4f}/{res['noise_16k'][m]['deg2']:.4f}" for m in ("M14", "M15", "M16", "MU")))
    le = res["latest_eval"]
    print(f"latest eval {le['step']}: " + ", ".join(f"{m} {le[m]:.4f}" for m in ("M16", "M15", "M14", "MU", "PTB", "WT")))
    for m, b in res["bands"].items():
        print(f"band {m}: sd(Δ) {b['sd_drop']:.4f}, ladder 5–95% union [{b['ladder_union'][0]:.4f}, {b['ladder_union'][1]:.4f}]"
              f" → band [{b['band'][0]:.4f}, {b['band'][1]:.4f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
