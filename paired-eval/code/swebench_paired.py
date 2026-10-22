#!/usr/bin/env python3
"""Paired vs unpaired SE for pairs of SWE-bench Verified systems, from public per-instance results.

SWE-bench Verified 上,同一批题的两个系统:按题配对的标准误比不配对的小多少。

数据:SWE-bench/experiments 仓库在提交 40f164d5b8 时公开的逐题结果(evaluation/verified/<system>/results/results.json
里的 resolved 列表),共 135 个系统。
对每一对系统,d_i = 系统 A 第 i 题是否解决 − 系统 B 是否解决(取值 −1、0、1),
  配对标准误 = sd(d) / sqrt(N),sd 用总体方差 (p10 + p01) − (p10 − p01)²;
  不配对标准误 = hypot(sqrt(pa(1−pa)/N), sqrt(pb(1−pb)/N)),也就是 Marin 评测政策(#10197)判胜负时用的形式。
两种分析:
  1. 水平相近的所有系统对:两者都解决 75–225 题(15–45%),相差不超过 50 题(10 个百分点);
  2. 同一模型家族的 10 对:同一底座的微调、同一模型的前后版本、同一模型换不同 agent 框架。
用法(只检出各系统的 results.json):
  git clone --filter=blob:none --no-checkout https://github.com/SWE-bench/experiments.git swe_exp
  cd swe_exp && git sparse-checkout init --no-cone
  git sparse-checkout set --no-cone '/evaluation/verified/*/results/results.json'
  git checkout 40f164d5b8f1d249bf95a6df8b74b577fd8e519d && cd -
  python swebench_paired.py swe_exp ../results
"""

import csv
import itertools
import json
import math
import os
import statistics as st
import sys

N = 500  # SWE-bench Verified 的题数;没提交的题按未解决算
K = 1.96 / math.sqrt(0.9)  # 政策的判胜负系数,含 1/sqrt(0.9) 的基础设施失败余量
FAMILY = {
    "Qwen2.5-Coder-32B-Instruct fine-tunes: SWE-agent-LM-32B vs Skywork-SWE-32B": ["20250511_sweagent_lm_32b", "20250616_Skywork-SWE-32B"],
    "Qwen3-Coder-30B-A3B-Instruct (OpenHands) vs an EntroPO fine-tune of it (R2E-Gym)": ["20250805_openhands-Qwen3-Coder-30B-A3B-Instruct", "20250901_entroPO_R2E_QwenCoder30BA3B"],
    "Lingma SWE-GPT 72B, two submissions": ["20240918_lingma-agent_lingma-swe-gpt-72b", "20241002_lingma-agent_lingma-swe-gpt-72b"],
    "Lingma SWE-GPT 7B, two submissions": ["20240918_lingma-agent_lingma-swe-gpt-7b", "20241002_lingma-agent_lingma-swe-gpt-7b"],
    "SWE-Fixer, two versions": ["20241128_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor_20241128",
                                "20250306_SWE-Fixer_Qwen2.5-7b-retriever_Qwen2.5-72b-editor"],
    "Devstral Small 2505 (OpenHands) vs 2507 (SWE-agent)": ["20250520_openhands_devstral_small", "20250725_sweagent_devstral_small_2507"],
    "GLM-4.5 vs GLM-4.6, Z.ai's own submissions": ["20250728_zai_glm4-5", "20250930_zai_glm4-6"],
    "Kimi K2 (OpenHands: 0711-preview; SWE-agent: kimi-k2-instruct; Lingxi: 0905-preview)": ["20250716_openhands_kimi_k2", "20250804_codesweep_sweagent_kimi_k2_instruct", "20251014_Lingxi_kimi_k2"],
}


def load_systems(checkout):
    root = os.path.join(checkout, "evaluation", "verified")
    out = {}
    for s in sorted(os.listdir(root)):
        p = os.path.join(root, s, "results", "results.json")
        if os.path.exists(p):
            out[s] = set(json.load(open(p)).get("resolved", []))
    return out


def pair_stats(A, B):
    pa, pb = len(A) / N, len(B) / N
    p10, p01 = len(A - B) / N, len(B - A) / N
    se_p = math.sqrt((p10 + p01 - (p10 - p01) ** 2) / N)
    se_u = math.hypot(math.sqrt(pa * (1 - pa) / N), math.sqrt(pb * (1 - pb) / N))
    return {"rate_a": pa, "rate_b": pb, "discordant": p10 + p01, "se_paired": se_p, "se_unpaired": se_u, "ratio": se_p / se_u}


def main(checkout, outdir):
    os.makedirs(outdir, exist_ok=True)
    ok = load_systems(checkout)

    # 1. 水平相近的所有系统对
    rows = []
    for a, b in itertools.combinations(sorted(ok), 2):
        na, nb = len(ok[a]), len(ok[b])  # 用整数题数比较,避免 0.1 的浮点误差
        if not (75 <= na <= 225 and 75 <= nb <= 225) or abs(na - nb) > 50:
            continue
        rows.append({"system_a": a, "system_b": b, **pair_stats(ok[a], ok[b])})
    with open(os.path.join(outdir, "swebench_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    r = sorted(x["ratio"] for x in rows)
    q = lambda f: r[int(f * (len(r) - 1))]
    summary = {
        "swebench_experiments_commit": "40f164d5b8f1d249bf95a6df8b74b577fd8e519d", "systems": len(ok), "pairs": len(rows),
        "ratio_median": st.median(r), "ratio_p10": q(0.1), "ratio_p90": q(0.9),
        "decisive_gap_500_unpaired_pts": 100 * K * st.median(x["se_unpaired"] for x in rows),
        "decisive_gap_500_paired_pts": 100 * K * st.median(x["se_paired"] for x in rows),
        "discordant_median": st.median(x["discordant"] for x in rows),
        "systems_in_pairs": len(set(x["system_a"] for x in rows) | set(x["system_b"] for x in rows)),
    }

    # 2. 同一模型家族
    fam = []
    for g, ms in FAMILY.items():
        for a, b in itertools.combinations(ms, 2):
            if a in ok and b in ok:
                fam.append({"family": g, "system_a": a, "system_b": b, **pair_stats(ok[a], ok[b])})
    with open(os.path.join(outdir, "swebench_family_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam[0]))
        w.writeheader()
        w.writerows(fam)
    fr = sorted(x["ratio"] for x in fam)
    summary.update({"family_pairs": len(fam), "family_ratio_min": fr[0], "family_ratio_median": st.median(fr), "family_ratio_max": fr[-1]})
    json.dump(summary, open(os.path.join(outdir, "swebench_summary.json"), "w"), indent=1)

    print(f"systems with results: {len(ok)}")
    print(f"similar-strength pairs: {len(rows)}; paired/unpaired SE ratio median {summary['ratio_median']:.2f} "
          f"(p10 {summary['ratio_p10']:.2f}, p90 {summary['ratio_p90']:.2f})")
    print(f"decisive gap on 500 tasks under the policy rule: unpaired {summary['decisive_gap_500_unpaired_pts']:.1f} pts, "
          f"paired {summary['decisive_gap_500_paired_pts']:.1f} pts")
    for x in fam:
        print(f"  {x['family'][:55]:55s} {x['rate_a']:5.1%} vs {x['rate_b']:5.1%}  ratio {x['ratio']:.2f}")
    print(f"family pairs: {len(fam)}; ratio {fr[0]:.2f}–{fr[-1]:.2f}, median {st.median(fr):.2f}")
    print(f"systems appearing in the similar-strength pairs: {len(set(x['system_a'] for x in rows) | set(x['system_b'] for x in rows))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
