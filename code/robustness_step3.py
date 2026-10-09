#!/usr/bin/env python3
"""第 3 步的稳健性检验:一次改一个设定,看官方 macro 终点的中位数和 80% 区间怎么变。
σ_ε 和 σ_eval 沿用主结果(results/step3/forecast_step3.json),只重算 hero 这一侧。

用法:
  python robustness_step3.py ../data/snapshot_2026-10-09 ../results/step3
"""

import json
import os
import sys

import numpy as np

import forecast_step3 as F
from reproduce import load


def run(lads, target, hp, hy, sigma_eps, sigma_eval, *, smooth_deg=3, raw=False, p_min=F.POST_SWAP_P,
        ladder_w=None, kappa=None):
    old = (F.SMOOTH_DEG, F.smooth, F.LADDER_W)
    F.SMOOTH_DEG = smooth_deg
    if raw:  # 不平滑:逐 5% 拟合的点之间线性插值
        F.smooth = lambda ys: (lambda p: np.interp(p, sorted(ys), [ys[k] for k in sorted(ys)]))
    if ladder_w is not None:
        F.LADDER_W = ladder_w
    p_now = float(hp[-1])
    keep = hp >= p_min
    rows = F.hero_table(lads, target, hp[keep], hy[keep], p_now)  # 窗口更窄时,上面已经把早期的点截掉
    end, _ = F.mc_predict(rows, p_now, sigma_eps, sigma_eval, 40_000, kappa_fixed=kappa)
    F.SMOOTH_DEG, F.smooth, F.LADDER_W = old
    q = F.quantiles(end)
    return q


def main(snap, outdir):
    long, runs = load(snap)
    lads = F.build_ladders(long, runs)
    main_res = json.load(open(os.path.join(outdir, "forecast_step3.json")))
    out = {}
    for target in F.TARGETS:
        hp, hy = F.hero_series(long, target)
        key = "macro16_direct" if target == "macro16" else target
        se, sv = main_res[key]["sigma_eps"], main_res[key]["sigma_eval"]
        variants = {
            "主设定(三次平滑、≥30% 窗口、GB200 0.75、κ~U[0,1])": {},
            "平滑改二次": {"smooth_deg": 2},
            "不平滑(逐点插值)": {"raw": True},
            "锚定窗口只用 ≥45%": {"p_min": 0.45},
            "只用 GB200": {"ladder_w": {"GB200": 1.0, "H100-new": 0.0}},
            "GB200 与 H100 各半": {"ladder_w": {"GB200": 0.5, "H100-new": 0.5}},
            "只用 H100-new": {"ladder_w": {"GB200": 0.0, "H100-new": 1.0}},
            "不漂移(κ=0)": {"kappa": 0.0},
            "漂移全部持续(κ≈1)": {"kappa": 1.0},
        }
        out[target] = {}
        for name, kw in variants.items():
            lw = kw.get("ladder_w")
            if lw is not None:
                kw = dict(kw)
                kw["ladder_w"] = {k: v for k, v in lw.items() if v > 0}
            q = run(lads, target, hp, hy, se, sv, **kw)
            out[target][name] = q
            label = "macro16(直接锚定)" if target == "macro16" else target
            print(f"{label:16} {name:<40} 中位数 {q['median']:.4f}  80% [{q['q10']:.4f}, {q['q90']:.4f}]  95% [{q['q025']:.4f}, {q['q975']:.4f}]")
        print()
    json.dump(out, open(os.path.join(outdir, "robustness.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
