#!/usr/bin/env python3
"""补充检查(2026-10-09):按 10/3 重拟合的原设置,只把 lm_head 的 FLOPs 去掉,看地板和 hero 终点怎么变。

第 2 步的诊断(diagnose_floor.py)用的是 H100 d768–d1536:"不含 lm_head"的算力取自预注册的算力表,而那张表没有 d512。
这里改用每个 rung 自己的算力:含 lm_head 的算力(就是重拟合 config 里的 source_compute)减去 6·d·V·tokens(V = 128,256),
设置和重拟合完全一样:H100 d512–d1536、两种配比、30%–100% 共 15 个进度点、一个地板在所有拟合间共用、loss 空间最小二乘。
不改动冻结的预测。

用法:
  python lmhead_exact_refit.py ../data/snapshot_2026-10-09T0440Z ../addenda
"""

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

from diagnose_floor import HERO_C_NOLM, hero_end, shared_floor
from explore_step3 import h100_ladder
from reproduce import REFIT_HERO_C, load, refit_data, total_sse

GAP_REF = 2.039  # 8/19 预注册的终点


def floor_range(groups, F, s, tol=1.10):
    """SSE 不超过最小值 tol 倍的地板范围(两侧各用 brentq 找边界)。"""
    lmin = min(p[2] for pts in groups.values() for p in pts)
    f = lambda x: total_sse(x, groups) - tol * s
    lo_grid = np.linspace(0.0, F, 200)
    lo_vals = [f(x) for x in lo_grid]
    lo = next((brentq(f, lo_grid[i], lo_grid[i + 1]) for i in range(len(lo_grid) - 1)
               if lo_vals[i] > 0 >= lo_vals[i + 1]), 0.0)
    hi_grid = np.linspace(F, lmin - 1e-4, 200)
    hi_vals = [f(x) for x in hi_grid]
    hi = next((brentq(f, hi_grid[i], hi_grid[i + 1]) for i in range(len(hi_grid) - 1)
               if hi_vals[i] <= 0 < hi_vals[i + 1]), lmin - 1e-4)
    return lo, hi


def main(snap, outdir):
    long, runs = load(snap)
    g_lm, cfg = refit_data(long, runs, snap)
    src = cfg["source_compute"]
    lad = {"new": h100_ladder(runs, "new"), "old": h100_ladder(runs, "old")}
    for tag, c in src.items():  # 我们的含 lm_head 算力必须和重拟合 config 一致
        w, mix = tag.split("-")
        if w != "d6144":
            assert abs(lad[mix][w]["C_lm"] / c - 1) < 1e-9, tag
    g_nolm = {k: [(w, lad[k[0]][w]["C_nolm"] * k[1], L) for (w, _, L) in pts] for k, pts in g_lm.items()}

    out = {"setting": "H100 d512-d1536, old and new mixes, 15 progress points (30%-100%), one shared floor, "
                      "least squares in loss space (the Oct 3 refit's method)",
           "compute_ratio_with_over_without_lm_head": {w: lad["new"][w]["C_lm"] / lad["new"][w]["C_nolm"] for w in lad["new"]},
           "hero_compute_ratio": REFIT_HERO_C / HERO_C_NOLM}
    for label, groups, hero_c in (("with_lm_head", g_lm, REFIT_HERO_C), ("without_lm_head", g_nolm, HERO_C_NOLM)):
        F, s = shared_floor(groups)
        lo, hi = floor_range(groups, F, s)
        out[label] = {"floor": F, "floor_range_sse_within_10pct": [lo, hi],
                      "hero_final_new_mix": hero_end(F, groups[("new", 1.0)], hero_c),
                      "hero_final_new_mix_over_floor_range": [hero_end(lo, groups[("new", 1.0)], hero_c),
                                                              hero_end(hi, groups[("new", 1.0)], hero_c)],
                      "hero_final_old_mix": hero_end(F, groups[("old", 1.0)], hero_c) if ("old", 1.0) in groups else None}
    gap = out["with_lm_head"]["hero_final_new_mix"] - GAP_REF
    closed = out["with_lm_head"]["hero_final_new_mix"] - out["without_lm_head"]["hero_final_new_mix"]
    out["gap_to_prereg"] = {"gap": gap, "closed_by_removing_lm_head": closed, "share": closed / gap}

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "2026-10-09_lm_head_exact_refit.json")
    json.dump(out, open(path, "w"), indent=1)
    for label in ("with_lm_head", "without_lm_head"):
        r = out[label]
        print(f"{label}: floor {r['floor']:.4f} (SSE ≤ 1.1x: {r['floor_range_sse_within_10pct'][0]:.4f}–"
              f"{r['floor_range_sse_within_10pct'][1]:.4f}); hero final (new) {r['hero_final_new_mix']:.4f} "
              f"({r['hero_final_new_mix_over_floor_range'][0]:.4f}–{r['hero_final_new_mix_over_floor_range'][1]:.4f}); "
              f"old {r['hero_final_old_mix']:.4f}")
    print("ratios:", {k: round(v, 3) for k, v in out["compute_ratio_with_over_without_lm_head"].items()},
          "hero", round(out["hero_compute_ratio"], 4))
    print("gap:", {k: round(v, 4) for k, v in out["gap_to_prereg"].items()})
    print("saved", path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
