#!/usr/bin/env python3
"""第 1 步的检查:快照抓全了没有,并回答阅读笔记第八节列的几个待核实问题。

用法:
  python check_snapshot.py ../data/snapshot_2026-10-09
"""

import json
import os
import sys

import numpy as np
import pandas as pd

TARGET = "eval_dropless/paloma/macro_loss"
HERO_STEPS = 390_251


def main(snap):
    runs = pd.read_csv(os.path.join(snap, "tables", "runs.csv"))
    long = pd.read_csv(os.path.join(snap, "tables", "wandb_long.csv"))
    train = pd.read_csv(os.path.join(snap, "tables", "train_sampled_long.csv"))
    manifest = json.load(open(os.path.join(snap, "manifest.json")))

    print("== 抓取概况 ==")
    print(f"runs={len(runs)}  long rows={len(long)}  errors={len(manifest['errors'])}")
    for e in manifest["errors"]:
        print("  ERROR", e)
    for r in manifest["wandb"]["runs"]:
        for line in r["log"]:
            print("  note", line)
    print("github:", manifest["github_issues"])
    print("external:", {k: v["status"] for k, v in manifest["external"].items()})

    print("\n== 1. hero 的总步数 ==")
    hero = runs[runs.category == "hero"]
    print(hero[["run", "state", "stop_after_steps", "summary_step", "ref_eval_rows"]].to_string(index=False))
    fc = runs[runs.category == "forecast"]
    print("forecast run 的 num_train_steps:", sorted(fc.num_train_steps.dropna().unique().tolist()))

    print("\n== 2. 阶梯的学习率调度(线性还是余弦)==")
    print("判据:衰减走到 25% 时,LR/峰值 线性 = 0.7625,余弦 = 0.861(min_lr_ratio 0.05)")
    for run in ["rav-ladder-d768-v2", "rav-ladder-d1024", "rav-ladder-d1536", "rav-ladder-d2048-v3"]:
        lr = train[(train.run == run) & (train.key == "optim/learning_rate")].sort_values("step")
        if lr.empty:
            print(f"  {run}: 没有学习率数据")
            continue
        stop = runs.loc[runs.run == run, "stop_after_steps"].iloc[0]
        peak = lr.value.max()
        warm_end = lr.loc[lr.value.idxmax(), "step"]
        out = []
        for frac in (0.25, 0.5, 0.75):
            s = warm_end + frac * (stop - warm_end)
            v = np.interp(s, lr.step, lr.value) / peak
            out.append(f"{int(frac * 100)}%→{v:.3f}")
        print(f"  {run}: peak={peak:.3g} at step {warm_end}, decay-progress " + ", ".join(out)
              + f", last={lr.value.iloc[-1] / peak:.3f}@{lr.step.iloc[-1]}")
    lr = train[(train.category == "hero") & (train.key == "optim/learning_rate")].sort_values("step")
    if not lr.empty:
        peak = lr.value.max()
        s25 = 3902 + 0.25 * (HERO_STEPS - 3902)
        print(f"  hero: peak={peak:.4g}; 按线性公式在 25% 衰减处应为 0.7625,实际插值="
              f"{np.interp(s25, lr.step, lr.value) / peak:.4f}(样本点 {len(lr)} 个,最后一步 {lr.step.max()})")

    print("\n== 3. 阶梯 rung 的评测点(看 5% 吸附)==")
    for run in ["rav-ladder-d768-v2", "rav-ladder-d1024", "rav-ladder-d1536", "rav-ladder-d2048-v3"]:
        ev = long[(long.run == run) & (long.key == TARGET)].sort_values("step")
        stop = runs.loc[runs.run == run, "stop_after_steps"].iloc[0]
        fr = (ev.step / stop).round(3).tolist()
        print(f"  {run}: {len(ev)} 个评测点, 进度 {fr[:3]} … {fr[-3:]}, 最后 {ev.value.iloc[-1]:.4f}")

    print("\n== 4. hero 主线的评测序列(交接与重叠)==")
    ev = long[(long.category == "hero") & (long.key == TARGET)].sort_values(["step", "run"])
    for run in hero.run:
        e = ev[ev.run == run]
        if len(e):
            print(f"  {run}: {len(e)} 点, step {e.step.min()}–{e.step.max()}, macro {e.value.iloc[0]:.4f} → {e.value.iloc[-1]:.4f}")
    dup = ev[ev.duplicated("step", keep=False)]
    if len(dup):
        print("  重叠的评测步(多个 run 在同一步都有评测):")
        print(dup[["step", "run", "value"]].to_string(index=False))
    last = ev.iloc[-1]
    print(f"  最新:step {last.step}({last.step / HERO_STEPS:.1%}), macro {last.value:.4f}")

    print("\n== 5. 团队预测 run 的内容 ==")
    for run in fc.run:
        sub = long[long.run == run]
        keys = sorted(sub.key.unique())
        print(f"  {run}: {len(sub)} 行, {len(keys)} 个 key, step {sub.step.min()}–{sub.step.max()}; 例: {keys[:4]}")
    for run in ["hero-dropless-refit-new-20261003-aa4pdwep", "hero-dropless-refit-old-20261003-phq0qbmf"]:
        meta = json.load(open(os.path.join(snap, "raw", "wandb", "forecast", run, "meta.json")))
        cfg = {k: (v.get("value") if isinstance(v, dict) and "value" in v else v) for k, v in meta["config"].items()}
        print(f"  {run} config:")
        for k, v in cfg.items():
            print(f"    {k} = {str(v)[:300]}")

    print("\n== 6. 67B 回测数据 ==")
    bt = runs[runs.category == "backtest_67b"]
    for _, r in bt.iterrows():
        sub = long[long.run == r.run]
        macro = sub[sub.key.str.endswith("paloma/macro_loss")]
        print(f"  {r.run}: state={r.state}, summary_step={r.summary_step}, eval keys={sub.key.nunique()}, "
              f"macro 点={len(macro)}" + (f", step {macro.step.min()}–{macro.step.max()}" if len(macro) else ""))


if __name__ == "__main__":
    main(sys.argv[1])
