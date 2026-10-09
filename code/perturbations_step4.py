#!/usr/bin/env python3
"""第 4 步:四类扰动的量化与处理规则(说明见 04_第4步扰动.md)。

1. 配比 / 冷却配比:同宽度下比较 GB200、H100-new、H100-old 从 57.7% 到 100% 的剩余降幅,
   判断第 3 步以 GB200 形状为主时要不要调整。
2. 上下文切换对训练的影响:用第 81,716 步换通信方式时"丢弃率 3.4e-2 → 7.8e-5,loss −0.0065 nat"
   估计 loss 对丢弃率的敏感度,再代入 #9615 测到的 16K 丢弃率。
3. 训练被缩短:团队规定延误时最多砍约 25% 的 token,并让学习率在新终点衰减到 5%。
   近似算法:新终点值 = 第 3 步的终点 + [100% 拟合在算力 r·C 处的值 − 在 C 处的值]。
4. PTB 第二次曝光:核对第一次曝光后多久恢复。

用法:
  python perturbations_step4.py ../data/snapshot_2026-10-09 ../results/step3 ../results/step4
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import forecast_step3 as F
from reproduce import load

P_NOW = None  # main() 里填
RATIOS = [0.80, 0.85, 0.90, 0.95, 1.00]


def remaining_drop_table(lads):
    rows = []
    for t in F.TARGETS:
        for lad in ("GB200", "H100-new", "H100-old"):
            for w in ("d512", "d768", "d1024", "d1536"):
                g = lads[lad]["grids"][t].get(w)
                if not g or not all(x in g for x in (0.55, 0.60, 0.80, 1.0)):
                    continue
                now = np.interp(P_NOW, [0.55, 0.60], [g[0.55], g[0.60]])
                batch = {"GB200": {"d768": 1024, "d1024": 2048, "d1536": 6144}}.get(lad, {}).get(w, 1024)
                rows.append({"target": t, "ladder": lad, "width": w, "batch": batch,
                             "drop_now_to_end": now - g[1.0], "drop_80_to_end": g[0.80] - g[1.0]})
    return pd.DataFrame(rows)


def drop_sensitivity():
    # W&B 报告 / #8506:第 81,716 步换成 ragged all-to-all,丢弃率 3.4e-2 → 7.8e-5,loss 平均 −0.0065 nat
    slope = 0.0065 / (3.4e-2 - 7.8e-5)
    # #9615(单机架):4K 1.9e-4、8K 8.3e-4、16K 2.9e-3;11 机架的线上丢弃率约 7e-5
    return {"nat_per_unit_drop": slope,
            "8K": slope * (8.3e-4 - 1.9e-4), "16K": slope * (2.9e-3 - 1.9e-4),
            "note": "单机架的丢弃率约为 11 机架的 2 倍,这里按单机架的增量算,偏保守"}


def fit_at_fraction(grids, rungs, widths, conv, E, p):
    pts = [(rungs[w][conv] * p, grids[w][p]) for w in widths if p in grids[w]]
    c = np.array([x[0] for x in pts])
    y = np.array([x[1] for x in pts])
    s, i = np.polyfit(np.log(c), np.log(y - E), 1)
    return lambda C: E + np.exp(i) * C ** s


def shortened(lads, long, runs, n=40_000):
    """训练被缩短到 r·T 并在新终点完成衰减时,终点值的近似分布。"""
    res = json.load(open(os.path.join(STEP3, "forecast_step3.json")))
    out = {}
    hp, hy = F.hero_series(long, "macro15")
    rows = F.hero_table(lads, "macro15", hp, hy, P_NOW)
    # 给每一行算 100% 拟合在 r·C 与 C 处的差
    for r in rows:
        lad = lads[r["ladder"]]
        grids = dict(lad["grids"]["macro15"])
        if r["corr"] is not None:
            grids["d2048"] = F.extend_d2048(grids["d2048"], r["corr"])
        law = fit_at_fraction(grids, lad["rungs"], list(lad["rungs"]), r["conv"], r["E"], 1.0)
        C = F.HERO[r["conv"]]
        r["shift"] = {ratio: float(law(C * ratio) - law(C)) for ratio in RATIOS}
    pt = res["macro16"]["ptb_model"]
    for ratio in RATIOS:
        mod = []
        for r in rows:
            m = dict(r)
            m["f1"] = r["f1"] + r["shift"][ratio]
            m["b"] = r["b"] * (ratio - P_NOW) / (1 - P_NOW)
            m["se_b"] = r["se_b"] * (ratio - P_NOW) / (1 - P_NOW)
            mod.append(m)
        end15, _ = F.mc_predict(mod, P_NOW, res["macro15"]["sigma_eps"], res["macro15"]["sigma_eval"], n)
        _, d_tr = F.sample_D(pt, n, P_NOW, [ratio])
        end16 = end15 + d_tr[ratio] / 16
        out[f"{ratio:.2f}"] = {"steps": int(round(ratio * F.HERO_STEPS)), "macro16": F.quantiles(end16), "macro15": F.quantiles(end15)}
    return out


def ptb_recovery(long):
    h = long[(long.category == "hero") & (long.key == F.PTB_KEY)].drop_duplicates("step").sort_values("step")
    h = h[(h.step >= 194_999) & (h.step <= 224_999)]
    return {int(s): round(float(v), 4) for s, v in zip(h.step, h.value)}


def main(snap, step3, outdir):
    global P_NOW, STEP3
    STEP3 = step3
    os.makedirs(outdir, exist_ok=True)
    long, runs = load(snap)
    lads = F.build_ladders(long, runs)
    hp, _ = F.hero_series(long, "macro15")
    P_NOW = float(hp[-1])

    rd = remaining_drop_table(lads)
    rd.to_csv(os.path.join(outdir, "remaining_drop_by_ladder.csv"), index=False)
    print("== 1. 同宽度的剩余降幅(57.7% → 100%;80% → 100%)==")
    print(rd[rd.target == "macro15"].round(4).to_string(index=False))

    ds = drop_sensitivity()
    print("\n== 2. 丢弃率对 loss 的影响 ==")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ds.items()})

    sh = shortened(lads, long, runs)
    print("\n== 3. 训练被缩短(在新终点完成衰减)时的终点 ==")
    for k, v in sh.items():
        m16, m15 = v["macro16"], v["macro15"]
        print(f"  {k}(第 {v['steps']:,} 步):官方 {m16['median']:.4f} [{m16['q10']:.4f}, {m16['q90']:.4f}]   15 子集 {m15['median']:.4f} [{m15['q10']:.4f}, {m15['q90']:.4f}]")

    pr = ptb_recovery(long)
    print("\n== 4. PTB 第一次曝光前后(第 209,272–209,669 步读到)==")
    print(pr)

    json.dump({"remaining_drop": rd.to_dict(orient="records"), "drop_sensitivity": ds, "shortened": sh, "ptb_first_exposure": pr},
              open(os.path.join(outdir, "perturbations_step4.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
