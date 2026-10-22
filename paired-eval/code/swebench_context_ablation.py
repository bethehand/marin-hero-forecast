#!/usr/bin/env python3
"""131k vs 65k context (no summarization) in Marin's SWE-bench Verified ablation (#10493): where the difference comes from.

Marin #10493 的 SWE-bench Verified 上下文消融:131k 和 65k(都不做摘要)的差别从哪里来。

数据:HF 数据集 open-athena/swebench-context-compaction-ablation-2026-10-22,固定在提交 21991381dc。
用 accepted-manifest.json(每个模型 × 每组的逐题得分)和 traces/<model>/<arm>.jsonl.zst(逐题轨迹)。
只用两组不做摘要的:65k-16k-no-summary 和 131k-16k-no-summary;四个模型:Qwen 3.6、Gemma 4、Nemotron Lightning、Snowball v0.1。
另外读 Marin 仓库里这次消融的补跑清单(configs/recovery/<model>/<arm>/manifest.json,固定在提交 58ff2c03aa),
统计原始运行保留了多少题、有多少题因为旧版运行时到了截止时间而重跑。

思路:两组的 Harbor 配置只有 max_input_tokens 不同(65,536 对 131,072),服务配置只有 max_model_len 不同,
所以一条轨迹在用到 65,536 个 token 之前,两组里的行为一样(前提是两组的服务速度差不多,见墙钟超时);
差别只能来自用到 65,536 以上的那些题("长"题)。
  65k 组的长题:因上下文用完退出(ContextLimit),或某次调用 prompt + completion 达到 65,536(在边界被截断);
  131k 组的长题:某次调用 prompt + completion 超过 65,536。
在两组共同的题上,逐题差 y131 − y65 = [y131·L131 − y65·L65](长题部分)+ [y131·(1−L131) − y65·(1−L65)](短题部分)。
两组在 65,536 以下行为一样时,短题部分期望为 0,只是噪声;长题部分是差别的无偏估计,方差更小。
短题部分接近 0、两组长题比例接近,是这个假设的检验(结果里都有)。墙钟超时(1,800 秒)会破坏这个假设:
慢的一组更容易超时,所以也单独统计了超时、每步耗时、每次调用的输出长度和输出速度。

需要 zstd 命令行工具(zstdcat)和网络(读补跑清单)。用法(先按 README 的 Reproduce 一节下载固定版本的文件,约 150 MB):
  python swebench_context_ablation.py <数据目录> ../results/swebench_context_ablation.json
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from datetime import datetime

MODELS = ("qwen36", "gemma4", "nemotron-lightning", "snowball-v0.1")
A65, A131 = "65k-16k-no-summary", "131k-16k-no-summary"
W = 65536
MARIN_SHA = "58ff2c03aa4e14ca4d8c109aaf286e9988135bae"
RECOVERY = ("https://raw.githubusercontent.com/marin-community/marin/" + MARIN_SHA +
            "/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/ablations/swebench_context_compaction/configs/recovery/{}/{}/manifest.json")


def check_hashes(data):
    """和 MANIFEST-*.json 里记录的 sha256 核对轨迹文件。"""
    for mo in MODELS:
        for a in json.load(open(os.path.join(data, f"MANIFEST-{mo}.json")))["arms"]:
            if a["arm"] in (A65, A131):
                p = os.path.join(data, a["file"])
                assert hashlib.sha256(open(p, "rb").read()).hexdigest() == a["sha256"], p


def when(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if isinstance(s, str) else s


def trials(path):
    """每题一行:得分、退出原因、是否超时、模型调用次数(api_calls,mini-swe-agent 的 step_limit 数的就是它,
    包括因为没有工具调用被拒绝的回复)、有效回复数、单次调用最大 prompt + completion、prompt 和 completion token 总数、
    墙钟秒数、模型时间(从开始到第一条回复,加上每条回复的时间戳减去上一条消息的时间戳)、单次调用最长秒数、
    最后一条消息之后到结束的空等秒数、是否在 GB200 上服务。"""
    out = {}
    raw = subprocess.run(["zstdcat", path], check=True, capture_output=True).stdout
    for line in raw.splitlines():
        row = json.loads(line)
        att = [a for a in row["attempts"] if "trajectory" in a]
        tr = att[-1]["trajectory"] if att else None
        ae = row.get("agent_execution") or {}
        start, end = when(ae.get("started_at")), when(ae.get("finished_at"))
        pc, p, c, model_s, longest, prev, last = [], [], [], 0.0, 0.0, start, None
        for m in (tr or {}).get("messages", []):
            ex = m.get("extra") or {}
            t = when(ex.get("timestamp"))
            if m.get("role") == "assistant":
                u = (ex.get("response") or {}).get("usage") or {}
                if u.get("prompt_tokens") is not None and u.get("completion_tokens") is not None:
                    p.append(u["prompt_tokens"])
                    c.append(u["completion_tokens"])
                    pc.append(u["prompt_tokens"] + u["completion_tokens"])
                if t is not None and prev is not None:
                    model_s += t - prev
                    longest = max(longest, t - prev)
            if t is not None:
                prev = last = t
        info = (tr or {}).get("info") or {}
        out[row["task_name"]] = {
            "reward": float(row["reward"]),
            "exit": info.get("exit_status") if tr else None,
            "timeout": row.get("exception_type") == "AgentTimeoutError",
            "api_calls": (info.get("model_stats") or {}).get("api_calls") or 0, "responses": len(pc),
            "max_pc": max(pc, default=0), "prompt_tokens": sum(p), "completion_tokens": sum(c),
            "seconds": end - start if start is not None and end is not None else None,
            "model_seconds": model_s, "longest_call": longest,
            "idle_after_last_message": end - last if end is not None and last is not None else None,
            "gb200": "gb200" in row["source_uri"],
        }
    return out


def mean_se(v):
    n = len(v)
    m = sum(v) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1) / n)


def median(v):
    v = sorted(v)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else None


def step(r):
    """每次模型调用的墙钟秒数(含命令执行)。"""
    return r["seconds"] / r["api_calls"] if r["seconds"] and r["api_calls"] else None


def quantile(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def slowdown(s, f, limit=1800.0, tasks=None):
    """服务变慢 f 倍:每题的模型时间乘 f,命令执行时间不变;超过 1,800 秒的题被截断。
    原本就超时的题不计入。返回 (被截断的题数, 其中原本解出的题数)。被截断的题仍按工作区评分,所以后者是上限。"""
    cut = [r for t, r in s.items() if (tasks is None or t in tasks) and not r["timeout"] and r["seconds"] is not None
           and r["model_seconds"] * f + (r["seconds"] - r["model_seconds"]) > limit]
    return len(cut), sum(r["reward"] for r in cut)


def arm_stats(s, long_):
    """一组的全部题(不限于共同题)。"""
    n = len(s)
    lg = [r for t, r in s.items() if long_[t]]
    fin = [r for r in lg if not r["timeout"]]
    to = [r for r in s.values() if r["timeout"]]
    by_hw = {hw: median([r["model_seconds"] / r["api_calls"] for r in s.values() if r["gb200"] == (hw == "gb200") and r["api_calls"] and not r["timeout"]])
             for hw in ("h100", "gb200")}
    pace = [step(r) for r in lg if step(r)]
    hung = [r for r in to if r["exit"] in ("Submitted", "ContextLimit")]
    return {
        # 步数按模型调用次数(api_calls)计:解出的题最多用了多少次;上限 L 会截断多少题、其中多少解出
        "max_calls_solved": max((r["api_calls"] for r in s.values() if r["reward"] == 1), default=None),
        "max_calls": max(r["api_calls"] for r in s.values()),
        "rejected_calls": sum(r["api_calls"] - r["responses"] for r in s.values()),
        "over_call_limit": {str(L): [sum(r["api_calls"] > L for r in s.values()), sum(r["reward"] for r in s.values() if r["api_calls"] > L)]
                            for L in (150, 200, 250, 300)},
        # 长题每次调用的墙钟秒数:中位数、90 和 95 分位;按这个速度跑 250 次调用要多少分钟
        "long_pace_seconds_per_call": [median(pace), quantile(pace, 0.9), quantile(pace, 0.95)],
        "minutes_for_250_calls": [x * 250 / 60 if x else None for x in (median(pace), quantile(pace, 0.9), quantile(pace, 0.95))],
        "long_share_slower_than_one_hour_for_250": sum(x > 3600 / 250 for x in pace) / len(pace) if pace else None,
        # 超时的题里,智能体其实已经退出(Submitted 或 ContextLimit)、之后空等到被杀的
        "timeouts_after_agent_exit": len(hung), "timeouts_after_agent_exit_solved": sum(r["reward"] for r in hung),
        "timeouts_after_agent_exit_idle_seconds": [min((r["idle_after_last_message"] for r in hung), default=None),
                                                    max((r["idle_after_last_message"] for r in hung), default=None)],
        # 超时的题里,有单次调用超过 5 分钟的
        "timeouts_with_a_call_over_300s": sum(r["longest_call"] > 300 for r in to),
        # 服务变慢 1.5、2、3 倍时被 1,800 秒截断的题数和损失的解出数
        "slowdown": {str(f): list(slowdown(s, f)) for f in (1.5, 2.0, 3.0)},
        # 按服务硬件分的每次调用模型时间中位数(同一模型同一组;不同作业跑的题不同)
        "median_model_seconds_per_call": by_hw,
        "n": n, "solve": sum(r["reward"] for r in s.values()) / n,
        "long": len(lg), "long_solved": sum(r["reward"] for r in lg), "long_timed_out": len(lg) - len(fin),
        "long_finished_solved": sum(r["reward"] for r in fin),
        "context_limit_exits": sum(r["exit"] == "ContextLimit" for r in s.values()),
        "timeouts": len(to), "timeouts_solved": sum(r["reward"] for r in to),
        "timeout_seconds": [min((r["seconds"] for r in to), default=None), max((r["seconds"] for r in to), default=None)],
        "median_seconds_per_step_long": median([step(r) for r in lg if step(r)]),
        "median_seconds_per_step_long_timed_out": median([step(r) for r in lg if r["timeout"] and step(r)]),
        "median_seconds_per_step_long_finished": median([step(r) for r in fin if step(r)]),
        "median_seconds_per_step_short": median([step(r) for t, r in s.items() if not long_[t] and not r["timeout"] and step(r)]),
        "median_completion_tokens_per_call_long": median([r["completion_tokens"] / r["responses"] for r in lg if r["responses"]]),
        "median_output_tokens_per_model_second_long": median([r["completion_tokens"] / r["model_seconds"] for r in lg if r["model_seconds"] > 0]),
        "mean_prompt_tokens_per_trial": sum(r["prompt_tokens"] for r in s.values()) / n,
        "served_on_gb200": sum(r["gb200"] for r in s.values()),
    }


def recovery(mo, arm, s):
    """补跑清单:原始运行保留的题数,因旧版运行时到截止时间而重跑的题,以及这些题最终的结果。"""
    m = json.load(urllib.request.urlopen(RECOVERY.format(mo, arm), timeout=120))
    dl = [t["task_name"] for t in m["observed_trials"] if t.get("recovery_reason") == "old_runtime_agent_deadline"]
    return {"retained_from_original_run": len(m["retained_tasks"]), "deadline_replays": len(dl),
            "replays_timed_out_again": sum(s[t]["timeout"] for t in dl), "replays_finished": sum(not s[t]["timeout"] for t in dl),
            "replays_finished_solved": sum(s[t]["reward"] for t in dl if not s[t]["timeout"])}


def main(data, out):
    check_hashes(data)
    man = json.load(open(os.path.join(data, "accepted-manifest.json")))
    reward = {(a["model"], a["arm"]): {x["task_name"]: float(x["reward"]) for x in a["trials"]} for a in man["arms"]}
    res = {"dataset": "open-athena/swebench-context-compaction-ablation-2026-10-22",
           "revision": "21991381dc1910473789b63b8bbde1353c0491d0", "marin_commit": MARIN_SHA, "models": {}}
    for mo in MODELS:
        a, b = trials(os.path.join(data, "traces", mo, A65 + ".jsonl.zst")), trials(os.path.join(data, "traces", mo, A131 + ".jsonl.zst"))
        # 轨迹里的得分和 accepted-manifest 一致
        assert {t: r["reward"] for t, r in a.items()} == reward[(mo, A65)] and {t: r["reward"] for t, r in b.items()} == reward[(mo, A131)], mo
        l65 = {t: r["exit"] == "ContextLimit" or r["max_pc"] >= W for t, r in a.items()}
        l131 = {t: r["max_pc"] > W for t, r in b.items()}
        com = sorted(set(a) & set(b))
        x, y = [b[t]["reward"] for t in com], [a[t]["reward"] for t in com]
        diff = mean_se([u - v for u, v in zip(x, y)])
        se_unpaired = math.hypot(mean_se(x)[1], mean_se(y)[1])
        long_part = mean_se([b[t]["reward"] * l131[t] - a[t]["reward"] * l65[t] for t in com])
        short_part = mean_se([b[t]["reward"] * (not l131[t]) - a[t]["reward"] * (not l65[t]) for t in com])
        p_long = mean_se([l131[t] - l65[t] for t in com])
        # 共同题上的长题数、其中解出的题数和超时数(长题部分 = (131k 解出 − 65k 解出) / 共同题数)
        lc = {"65k": sum(l65[t] for t in com), "131k": sum(l131[t] for t in com),
              "solved_65k": sum(a[t]["reward"] for t in com if l65[t]), "solved_131k": sum(b[t]["reward"] for t in com if l131[t]),
              "timed_out_131k": sum(b[t]["timeout"] for t in com if l131[t]),
              "timeouts_65k_all_common": sum(a[t]["timeout"] for t in com), "timeouts_131k_all_common": sum(b[t]["timeout"] for t in com),
              # 服务变慢时的截断,限于共同题
              "slowdown_65k": {str(f): list(slowdown(a, f, tasks=set(com))) for f in (1.5, 2.0, 3.0)},
              "slowdown_131k": {str(f): list(slowdown(b, f, tasks=set(com))) for f in (1.5, 2.0, 3.0)}}
        e = {"common_tasks": len(com), "on_common_tasks": lc,
             "diff": {"mean": diff[0], "se_paired": diff[1], "se_unpaired": se_unpaired, "ratio": diff[1] / se_unpaired},
             "long_part": {"mean": long_part[0], "se": long_part[1]},
             "short_part": {"mean": short_part[0], "se": short_part[1]},
             "long_share_diff": {"mean": p_long[0], "se": p_long[1]},
             A65: {**arm_stats(a, l65), "recovery": recovery(mo, A65, a)},
             A131: {**arm_stats(b, l131), "recovery": recovery(mo, A131, b)}}
        res["models"][mo] = e
        s65, s131 = e[A65], e[A131]
        print(f"== {mo}: {len(com)} common tasks")
        print(f"   131k − 65k {diff[0]:+.4f} (paired SE {diff[1]:.4f}, unpaired {se_unpaired:.4f}, ratio {diff[1] / se_unpaired:.2f})"
              f" = long part {long_part[0]:+.4f} (SE {long_part[1]:.4f}) + short part {short_part[0]:+.4f} (SE {short_part[1]:.4f})")
        print(f"   common tasks: long {lc['65k']} / {lc['131k']}, solved {lc['solved_65k']:.0f} / {lc['solved_131k']:.0f}, long timed out at 131k {lc['timed_out_131k']};"
              f" timeouts {lc['timeouts_65k_all_common']} → {lc['timeouts_131k_all_common']}; long share 131k − 65k {p_long[0]:+.3f} (SE {p_long[1]:.3f})")
        print(f"   all trials: long 65k {s65['long']}/{s65['n']}, 131k {s131['long']}/{s131['n']} (timed out {s131['long_timed_out']},"
              f" finished {s131['long'] - s131['long_timed_out']} of which solved {s131['long_finished_solved']:.0f}); timeouts {s65['timeouts']} → {s131['timeouts']}")
        print(f"   s/step: long@131k {s131['median_seconds_per_step_long']:.1f} (timed out {s131['median_seconds_per_step_long_timed_out'] or float('nan'):.1f},"
              f" finished {s131['median_seconds_per_step_long_finished']:.1f}); short {s65['median_seconds_per_step_short']:.1f} → {s131['median_seconds_per_step_short']:.1f}")
        print(f"   long@131k: {s131['median_completion_tokens_per_call_long']:.0f} completion tokens/call, {s131['median_output_tokens_per_model_second_long']:.0f} tokens per model-second"
              f" | prompt tokens/trial {s65['mean_prompt_tokens_per_trial'] / 1e6:.2f}M → {s131['mean_prompt_tokens_per_trial'] / 1e6:.2f}M | GB200 {s65['served_on_gb200']} / {s131['served_on_gb200']}")
        print(f"   model calls: max solved {s65['max_calls_solved']} / {s131['max_calls_solved']}, max {s65['max_calls']} / {s131['max_calls']},"
              f" rejected {s65['rejected_calls']} / {s131['rejected_calls']}; over 250 calls at 65k {s65['over_call_limit']['250']},"
              f" at 131k {s131['over_call_limit']['250']}; over 300 at 131k {s131['over_call_limit']['300']}")
        print(f"   slowdown x1.5/x2/x3 (cut, of which solved): all trials 65k {s65['slowdown']} 131k {s131['slowdown']};"
              f" common tasks 65k {lc['slowdown_65k']} 131k {lc['slowdown_131k']}")
        pz, mz = s131["long_pace_seconds_per_call"], s131["minutes_for_250_calls"]
        if pz[0]:
            print(f"   131k long pace s/call median/p90/p95 {pz[0]:.1f}/{pz[1]:.1f}/{pz[2]:.1f} → 250 calls {mz[0]:.0f}/{mz[1]:.0f}/{mz[2]:.0f} min;"
                  f" share slower than 14.4 s/call {s131['long_share_slower_than_one_hour_for_250']:.0%}")
        for arm, sx in ((A65, s65), (A131, s131)):
            if sx["timeouts"]:
                print(f"   {arm[:4]} timeouts {sx['timeouts']}: after agent exit {sx['timeouts_after_agent_exit']} (solved {sx['timeouts_after_agent_exit_solved']:.0f},"
                      f" idle {sx['timeouts_after_agent_exit_idle_seconds']}); with a call over 300 s {sx['timeouts_with_a_call_over_300s']}")
        print(f"   model s/call by hardware: 65k {s65['median_model_seconds_per_call']} 131k {s131['median_model_seconds_per_call']}")
        print(f"   recovery: original-run trials kept {s65['recovery']['retained_from_original_run']} / {s131['recovery']['retained_from_original_run']};"
              f" deadline replays {s65['recovery']['deadline_replays']} / {s131['recovery']['deadline_replays']}"
              f" (131k: timed out again {s131['recovery']['replays_timed_out_again']}, finished {s131['recovery']['replays_finished']}, solved {s131['recovery']['replays_finished_solved']:.0f})")
    for k in ("diff", "long_part"):
        v = [res["models"][mo][k] for mo in MODELS]
        m = sum(z["mean"] for z in v) / len(v)
        se = math.sqrt(sum(z.get("se_paired", z.get("se")) ** 2 for z in v)) / len(v)
        res[f"pooled_{k}"] = {"mean": m, "se": se, "ci95": [m - 1.96 * se, m + 1.96 * se]}
        print(f"equal-weight mean over the four models, {k}: {m:+.4f} (SE {se:.4f}, 95% CI [{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}])")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
