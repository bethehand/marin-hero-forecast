#!/usr/bin/env python3
"""Ties, near-threshold ties and the pick under scaled thresholds in Marin's published Snowball tournaments.

Marin 公开的两轮 Snowball 选模型比赛:平局有多少、离门槛多近、门槛收紧后谁领先。

数据:penfever 在 #10199(v0)和 #10283(v0.1)公开的 gist,里面的 snowball_benchmark_pair_outcomes.csv
有每一对候选模型在每个评测上的分差和判胜负门槛(1.96 × hypot(adjusted SEM)),snowball_selection.json 有候选名单。
没有逐题数据,所以这里不能算真正的配对标准误,只把门槛乘一个统一的系数 r,看结果对门槛有多敏感。
两种缩放:
  all:   所有评测的门槛都乘 r;
  single:只缩放每题一次的 19 个评测;TBLite、TB2.1、AIME24、OlympiadBench 每题 3–10 次,
         现在的 SEM 大多把每次都当独立样本,按题配对后门槛不一定更小,所以保持不变。
计分和 pairwise.py 相同(赢 1、平 0.5、输 0;对每个对手取胜率,再对所有对手取平均),
用精确分数计算;并列时按政策的规则先比合并胜率,再按模型 ID 排。r = 1 时复现公布的胜率(误差 1e-15 以内)。
用法(先 git clone 两个 gist 并检出固定版本):
  git clone https://gist.github.com/penfever/8baa8321ad1d08ce514be90a25a5c0d2.git v0
  git -C v0 checkout 69708d2f38b78389f8f7d20cbc504190f7e9095f
  git clone https://gist.github.com/penfever/b7deea3a2eaec1642a56c0d67ba1e41a.git v01
  git -C v01 checkout 191228e26bc3c2ef1a3c2496a7fd192d10f3c1d8
  python tournament_sensitivity.py v0 v01 ../results/tournament_sensitivity.json
"""

import collections
import csv
import json
import math
import os
import sys
from fractions import Fraction

MULTI_TRIAL = frozenset({"ot-tblite-recovery", "tb2-recovery", "aime24", "olympiadbench"})


def load(g):
    rows = list(csv.DictReader(open(os.path.join(g, "snowball_benchmark_pair_outcomes.csv"))))
    cands = json.load(open(os.path.join(g, "snowball_selection.json")))["candidates"]
    pub = {r["model"]: float(r["mean_pairwise_win_rate"])
           for r in csv.DictReader(open(os.path.join(g, "snowball_pairwise_summary.csv")))}
    return rows, cands, pub


def ratio(x):
    """|分差| / 门槛;门槛为 0 时,分差非 0 记为无穷大。"""
    d, t = abs(float(x["score_difference"])), float(x["difference_95_threshold"])
    return (math.inf if d > 0 else 0.0) if t == 0 else d / t


def ranking(rows, cands, r, mode):
    """返回 [(model, mean_win_rate, pooled_win_rate)],按政策排序:均值降序、合并胜率降序、模型 ID 升序。"""
    pts = collections.defaultdict(lambda: [0, 0])  # (model, opp) -> [半分数, 评测数]
    for x in rows:
        d, t = float(x["score_difference"]), float(x["difference_95_threshold"])
        if mode == "all" or x["benchmark"] not in MULTI_TRIAL:
            t *= r
        a, b = x["left_model"], x["right_model"]
        pa = 2 if d > t else 0 if d < -t else 1
        pts[(a, b)][0] += pa
        pts[(a, b)][1] += 1
        pts[(b, a)][0] += 2 - pa
        pts[(b, a)][1] += 1
    out = []
    for m in cands:
        opp = [pts[(m, o)] for o in cands if o != m]
        mean = sum(Fraction(p, 2 * n) for p, n in opp) / len(opp)
        pooled = Fraction(sum(p for p, _ in opp), 2 * sum(n for _, n in opp))
        out.append((m, mean, pooled))
    return sorted(out, key=lambda t: (-t[1], -t[2], t[0]))


def state(rank):
    """领先者;前两名均值完全相等时标为 exact tie(此时由合并胜率和模型 ID 决定)。"""
    return rank[0][0] if rank[0][1] != rank[1][1] else "exact tie: " + " | ".join(sorted([rank[0][0], rank[1][0]]))


def main(out, *gists):
    res = {}
    for g in gists:
        rows, cands, pub = load(g)
        base = ranking(rows, cands, 1.0, "all")
        assert all(abs(float(m) - pub[name]) < 1e-15 for name, m, _ in base), f"{g}: r = 1 does not reproduce the published win rates"
        name = os.path.basename(os.path.normpath(g))
        entry = {"candidates": len(cands), "published_pick": base[0][0], "published_runner_up": base[1][0],
                 "pick_rate": float(base[0][1]), "runner_up_rate": float(base[1][1])}
        t1, t2 = base[0][0], base[1][0]
        hh = [x for x in rows if {x["left_model"], x["right_model"]} == {t1, t2}]
        entry["top_two_head_to_head"] = {"benchmarks": len(hh), "ties": sum(x["outcome"] == "tie" for x in hh)}
        for mode in ("all", "single"):
            sub = rows if mode == "all" else [x for x in rows if x["benchmark"] not in MULTI_TRIAL]
            ties = [x for x in sub if x["outcome"] == "tie"]
            changes, prev = [], None
            for i in range(10000, 4999, -1):  # r 从 1.0000 到 0.5000,步长 0.0001
                s = state(ranking(rows, cands, i / 10000, mode))
                if s != prev:
                    changes.append({"from_r": i / 10000, "leader": s})
                    prev = s
            entry[mode] = {"comparisons": len(sub), "ties": len(ties),
                           "ties_with_ratio_at_least": {f"{lo:.2f}": sum(1 for x in ties if ratio(x) >= lo) for lo in (0.5, 0.6, 0.71, 0.8)},
                           "leader_changes": changes}
        res[name] = entry
        print(f"== {name}: {len(cands)} candidates; pick {t1.split('/')[-1]} ({float(base[0][1]):.4f}), "
              f"runner-up {t2.split('/')[-1]} ({float(base[1][1]):.4f}); top two tie on {entry['top_two_head_to_head']['ties']} "
              f"of {entry['top_two_head_to_head']['benchmarks']} benchmarks")
        for mode in ("all", "single"):
            e = entry[mode]
            n = e["comparisons"]
            near = e["ties_with_ratio_at_least"]
            print(f"   [{mode}] {n} comparisons, ties {e['ties'] / n:.1%}, ties >= 0.71x {near['0.71'] / n:.1%}, >= 0.5x {near['0.50'] / n:.1%}")
            for c in e["leader_changes"]:
                print(f"      from r = {c['from_r']:.4f}: {c['leader'].replace('open-athena/', '')}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[-1], *sys.argv[1:-1])
