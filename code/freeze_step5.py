#!/usr/bin/env python3
"""第 5 步:用最新快照重跑第 3、4 步,轨迹加密到每 1%,输出冻结用的表、英文图和 sha256 清单。

用法:
  python freeze_step5.py ../data/snapshot_<时间> ../results/step5
之后写好预注册文档,再运行:
  python freeze_step5.py --hash-only ../results/step5 ../05_preregistration.md
"""

import hashlib
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import forecast_step3 as F
import perturbations_step4 as P4
from reproduce import load

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ["fetch_snapshot.py", "check_snapshot.py", "reproduce.py", "diagnose_floor.py", "explore_step3.py",
              "forecast_step3.py", "robustness_step3.py", "perturbations_step4.py", "freeze_step5.py"]
TEAM_EN = {"pre-registered (Aug 19)": 2.039, "swap-adjusted (Sep 14)": 2.02685, "refit, new mix (Oct 3)": 2.1001}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_traj_csv(traj, path, p_now):
    with open(path, "w") as f:
        f.write("fraction,step,q025,q10,median,q90,q975\n")
        for p, q in traj.items():
            pf = float(p)
            if pf + 1e-9 < p_now:
                continue
            f.write(f"{pf:.2f},{int(round(pf * F.HERO_STEPS))},{q['q025']:.4f},{q['q10']:.4f},{q['median']:.4f},{q['q90']:.4f},{q['q975']:.4f}\n")


def plot_en(res, long, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, title in ((axes[0], "macro16", "eval_dropless/paloma/macro_loss (16 subsets, primary)"),
                           (axes[1], "macro15", "15-subset macro without PTB (companion)")):
        hp, hy = F.hero_series(long, "macro16" if key == "macro16" else "macro15")
        tr = res[key]["trajectory"]
        ps = np.array([float(p) for p in tr]) * 100
        ax.plot(hp * 100, hy, "-", color="black", lw=1.4, label="hero, observed")
        ax.fill_between(ps, [tr[p]["q025"] for p in tr], [tr[p]["q975"] for p in tr], color="#1f77b4", alpha=0.18, label="95% interval")
        ax.fill_between(ps, [tr[p]["q10"] for p in tr], [tr[p]["q90"] for p in tr], color="#1f77b4", alpha=0.35, label="80% interval")
        ax.plot(ps, [tr[p]["median"] for p in tr], "-", color="#1f77b4", lw=2, label="median")
        if key == "macro16":
            for (name, v), col in zip(TEAM_EN.items(), ("crimson", "darkorange", "seagreen")):
                ax.plot(100, v, "D", color=col, ms=8, label=f"team {name}: {v:.3f}")
        lo = min(hy[hp > 0.25].min(), min(tr[p]["q025"] for p in tr)) - 0.02
        ax.set_ylim(lo, hy[hp > 0.25].max() + 0.01)
        ax.set_xlim(25, 102)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training progress (%)")
        ax.set_ylabel("dropless Paloma loss (4K packing)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="upper right")
    step = res["macro16"]["data_until_step"]
    fig.suptitle(f"Anchored forecast of the 535B-A23B hero run's final loss (data through step {step:,})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def run(snap, outdir):
    os.makedirs(outdir, exist_ok=True)
    long, _ = load(snap)
    hp, _ = F.hero_series(long, "macro15")
    p_now = float(hp[-1])
    start = math.ceil(p_now * 100 - 1e-9) / 100
    F.P_OUT = [round(x, 2) for x in np.arange(start, 1.0 + 1e-9, 0.01)]
    if F.P_OUT[-1] != 1.0:
        F.P_OUT.append(1.0)
    F.main(snap, outdir)
    P4.main(snap, outdir, outdir)
    res = json.load(open(os.path.join(outdir, "forecast_step3.json")))
    write_traj_csv(res["macro16"]["trajectory"], os.path.join(outdir, "trajectory_official_macro16.csv"), p_now)
    write_traj_csv(res["macro15"]["trajectory"], os.path.join(outdir, "trajectory_macro15.csv"), p_now)
    plot_en(res, long, os.path.join(outdir, "forecast_en.png"))
    man = json.load(open(os.path.join(snap, "manifest.json")))
    summary = {
        "data_through_step": res["macro16"]["data_until_step"], "fraction": res["macro16"]["p_now"],
        "snapshot": os.path.relpath(snap, os.path.dirname(HERE)), "snapshot_fetched_utc": man["finished_utc"],
        "official_macro16": {"end": res["macro16"]["end"], "probs": res["macro16"]["probs"], "team_percentile": res["macro16"]["team_percentile"]},
        "macro15": {"end": res["macro15"]["end"], "probs": res["macro15"]["probs"]},
        "direct_macro16_variant": res["macro16_direct"]["end"],
        "shortened_horizon": json.load(open(os.path.join(outdir, "perturbations_step4.json")))["shortened"],
    }
    json.dump(summary, open(os.path.join(outdir, "forecast_summary.json"), "w"), ensure_ascii=False, indent=1)
    e = summary["official_macro16"]["end"]
    print(f"\n== 冻结用结果 == 数据截至第 {summary['data_through_step']:,} 步({summary['fraction']:.4f})")
    print(f"官方 macro:中位数 {e['median']:.4f};80% [{e['q10']:.4f}, {e['q90']:.4f}];95% [{e['q025']:.4f}, {e['q975']:.4f}]")


def hash_only(outdir, prereg):
    root = os.path.dirname(HERE)
    files = [os.path.join(HERE, f) for f in CODE_FILES]
    files += [os.path.join(outdir, f) for f in sorted(os.listdir(outdir)) if f.endswith((".json", ".csv", ".png", ".npz"))
              and f != "freeze_manifest.json"]
    files.append(os.path.abspath(prereg))
    summ = json.load(open(os.path.join(outdir, "forecast_summary.json")))
    snap_manifest = os.path.join(root, summ["snapshot"], "manifest.json")
    files.append(snap_manifest)
    out = {"frozen_files": {os.path.relpath(p, root): sha256(p) for p in files}}
    json.dump(out, open(os.path.join(outdir, "freeze_manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--hash-only":
        hash_only(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
