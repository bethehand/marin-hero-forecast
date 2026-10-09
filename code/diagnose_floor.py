#!/usr/bin/env python3
"""第 2 步的诊断:2.04(预注册)和 2.10(10/3 重拟合)的分歧从哪来。

三项检查都用重拟合的方法(loss 空间最小二乘、一个下界 F 在所有进度点和配比间共用、30%–100% 共 15 个进度点):
  1. 换阶梯:H100 阶梯 vs GB200 预注册阶梯(d2048 只用实测到 80% 的点,不外推)。
  2. 换宽度:H100 阶梯去掉 d512。
  3. 换 C 的口径:H100 阶梯 d768–d1536,C 用含 lm_head 的(重拟合原口径)和不含 lm_head 的(预注册口径)各算一次。
     两组阶梯在这三个宽度上的 token 数相同,只是 FLOPs 的记法不同。

用法:
  python diagnose_floor.py ../data/snapshot_2026-10-09 > ../results/step2_diagnosis.txt
"""

import sys

from scipy.optimize import minimize_scalar

from reproduce import PRE_FLOPS, PRE_RUNS, REFIT_FRACTIONS, REFIT_HERO_C, fit_group, load, prereg_grid, refit_data, total_sse

HERO_C_NOLM = PRE_FLOPS["d6144"][0] * PRE_FLOPS["d6144"][1]


def shared_floor(groups):
    lmin = min(p[2] for pts in groups.values() for p in pts)
    r = minimize_scalar(lambda F: total_sse(F, groups), bounds=(0.0, lmin - 1e-4), method="bounded", options={"xatol": 1e-7})
    return r.x, r.fun


def hero_end(F, pts, hero_c):
    A, alpha, _ = fit_group(F, pts)
    return F + A * hero_c ** (-alpha)


def main(snap):
    long, runs = load(snap)
    g_h100, cfg = refit_data(long, runs, snap)
    src = cfg["source_compute"]

    print("== 1. 换阶梯 ==")
    F, s = shared_floor(g_h100)
    print(f"H100 d512–d1536(含 lm_head):F={F:.4f};E 固定 1.5 时 SSE 是最优的 {total_sse(1.5, g_h100) / s:.2f} 倍;"
          f"hero 终点(new)={hero_end(F, g_h100[('new', 1.0)], REFIT_HERO_C):.4f}")
    g_gb = {}
    for r, name in PRE_RUNS.items():
        grid = prereg_grid(long, name, int(runs.loc[name, "stop_after_steps"]))
        for p in REFIT_FRACTIONS:
            if p in grid:
                g_gb.setdefault(("gb200", p), []).append((r, PRE_FLOPS[r][0] * PRE_FLOPS[r][1] * p, grid[p]))
    F, s = shared_floor(g_gb)
    print(f"GB200 d768–d2048(不含 lm_head;d2048 只到 80%):F={F:.4f};"
          f"E=1.5 时 SSE 是最优的 {total_sse(1.5, g_gb) / s:.2f} 倍,E=1.858 时 {total_sse(1.858, g_gb) / s:.2f} 倍")
    for F0 in (F, 1.5, 1.858):
        print(f"   F={F0:.4f} → hero 终点 {hero_end(F0, g_gb[('gb200', 1.0)], HERO_C_NOLM):.4f}")

    print("\n== 2. 换宽度 ==")
    g = {k: [p for p in v if p[0] != "d512"] for k, v in g_h100.items()}
    g = {k: v for k, v in g.items() if len(v) >= 3}
    F, _ = shared_floor(g)
    print(f"H100 d768–d1536(含 lm_head):F={F:.4f},组数 {len(g)}")

    print("\n== 3. 换 C 的口径(H100 d768–d1536)==")
    nolm = {w: PRE_FLOPS[w][0] * PRE_FLOPS[w][1] for w in ("d768", "d1024", "d1536")}
    print("含 lm_head / 不含 lm_head:", {w: round(src[f"{w}-new"] / nolm[w], 3) for w in nolm},
          f"hero {REFIT_HERO_C / HERO_C_NOLM:.4f}")
    for label, conv, hero_c in (("含 lm_head", None, REFIT_HERO_C), ("不含 lm_head", nolm, HERO_C_NOLM)):
        g = {}
        for (mix, p), pts in g_h100.items():
            q = [(w, (conv[w] * p if conv else c), L) for (w, c, L) in pts if w != "d512"]
            if len(q) >= 3:
                g[(mix, p)] = q
        F, _ = shared_floor(g)
        print(f"{label}:F={F:.4f},hero 终点(new)={hero_end(F, g[('new', 1.0)], hero_c):.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
