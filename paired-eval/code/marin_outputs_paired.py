#!/usr/bin/env python3
"""Paired vs unpaired SE per benchmark, and task-level vs per-sample SE on AIME24, from Marin's published per-sample outputs.

用 Marin 自己公开的逐题输出,按评测算"按题配对的标准误 / 不配对的标准误",以及 AIME24 多次采样时按题算和按样本算的差别。

数据:HF 数据集 marin-community/grug-67b-a2b-snowball-post-sft-evals,固定在提交 ad03b3669e(2026-07-25)。
里面有 7 个模型(Grug-67B-A2B SFT step630,以及 DeepSeek-V2-Lite-Chat、Ling-lite、Moonlight-16B-A3B、
Qwen3-30B-A3B-Thinking-2507、Qwen3.5-35B-A3B、Qwen3-Next-80B-A3B-Thinking)的逐题 samples parquet。
逐题得分的取法(都和公布的汇总分数核对过):
  ARC-C、BoolQ、WinoGrande、MATH500:correct 列;
  GSM8K:每题两行,取第二行(flexible-extract),和公布的 flexible-extract 分数完全一致;
  GPQA Diamond:correct 列在导出时一律为 True,改用 extracted 是否等于 target_text(每题只存了 3 次中的 1 次);
  AIME24:correct 列同样一律为 True,改用 extracted 和 doc 里的 expected_answer 比较;3 个种子各存 1 次,共 3 次/题。
HumanEval+、MBPP+、IFEval 的逐题得分没导出,要重新执行代码或检查指令,这里不算。
对每一对模型,按题取平均后:
  配对标准误 = sd(x_t − y_t) / sqrt(n),不配对标准误 = hypot(sd(x_t), sd(y_t)) / sqrt(n)。
需要 pandas 和 pyarrow。用法:
  python marin_outputs_paired.py <缓存目录> ../results/marin_outputs_paired.json
"""

import hashlib
import itertools
import json
import math
import os
import re
import statistics as st
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd

REPO = "marin-community/grug-67b-a2b-snowball-post-sft-evals"
REV = "ad03b3669eec9b583bb35e3790c989b4101763f4"
TASKS = ("arc_challenge", "boolq", "winogrande", "MATH500", "gsm8k", "GPQADiamond", "AIME24")
CLOSE = 0.15  # "水平相近":两个模型的准确率相差不超过 15 个百分点


def list_files():
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REV}/v2026-07-24/artifacts?recursive=true"
    files = []
    while url:
        resp = urllib.request.urlopen(url, timeout=120)
        files += [x for x in json.loads(resp.read()) if x.get("type") == "file"]
        m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link") or "")
        url = m.group(1) if m else None
    return files


def task_of(path):
    return re.sub(r".*/samples_(.+?)_2026.*", r"\1", path)


def fetch(cache, f):
    out = os.path.join(cache, hashlib.sha1(f["path"].encode()).hexdigest()[:16] + ".parquet")
    if not (os.path.exists(out) and os.path.getsize(out) == f["size"]):
        for _ in range(3):
            try:
                data = urllib.request.urlopen(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{f['path']}", timeout=300).read()
                open(out, "wb").write(data)
                break
            except Exception:
                continue
    assert os.path.getsize(out) == f["size"], f["path"]
    if "lfs" in f:  # 和 LFS 记录的 sha256 核对
        assert hashlib.sha256(open(out, "rb").read()).hexdigest() == f["lfs"]["oid"], f["path"]
    return out


def norm(s):
    s = str(s).strip().strip("$").replace(",", "")
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def score(task, df):
    """返回 {doc_id: 0/1}。"""
    if task == "gsm8k":
        df = df.drop_duplicates("doc_id", keep="last")
        c = df.correct.astype(float)
    elif task == "GPQADiamond":
        c = (df.extracted.astype(str).str.strip() == df.target_text.astype(str).str.strip()).astype(float)
    elif task == "AIME24":
        c = (df.extracted.map(norm) == df.doc.map(lambda d: norm(json.loads(d)["expected_answer"]))).astype(float)
    else:
        df = df.drop_duplicates("doc_id", keep="first")
        c = df.correct.astype(float)
    return dict(zip(df.doc_id.astype(str), c))


def main(cache, out):
    os.makedirs(cache, exist_ok=True)
    files = [f for f in list_files() if "/samples_" in f["path"] and f["path"].endswith(".parquet") and task_of(f["path"]) in TASKS]
    data = defaultdict(lambda: defaultdict(dict))  # task -> model -> run -> {doc: score}
    for f in files:
        parts = f["path"].split("/")
        model = parts[2].replace("penfever-", "").replace("marin-community-", "")  # 同一个 Grug checkpoint 分在两个命名空间
        data[task_of(f["path"])][model][parts[5]] = score(task_of(f["path"]), pd.read_parquet(fetch(cache, f)))
    res = {"dataset": REPO, "revision": REV, "close_pair_gap": CLOSE, "tasks": {}}
    for task in TASKS:
        models = sorted(data[task])
        per, trials = {}, {}
        for mo in models:
            runs = data[task][mo]
            docs = sorted(set.intersection(*[set(r) for r in runs.values()]))
            trials[mo] = np.array([[runs[r][d] for r in sorted(runs)] for d in docs])
            per[mo] = dict(zip(docs, trials[mo].mean(axis=1)))
        pairs = []
        for a, b in itertools.combinations(models, 2):
            docs = sorted(set(per[a]) & set(per[b]))
            x, y, n = np.array([per[a][d] for d in docs]), np.array([per[b][d] for d in docs]), len(docs)
            se_u = math.hypot(x.std(ddof=1), y.std(ddof=1)) / math.sqrt(n)
            if se_u > 0:
                pairs.append({"a": a, "b": b, "n": n, "acc_a": float(x.mean()), "acc_b": float(y.mean()),
                              "ratio": float((x - y).std(ddof=1) / math.sqrt(n) / se_u)})
        close = [p["ratio"] for p in pairs if abs(p["acc_a"] - p["acc_b"]) <= CLOSE]
        e = {"models": len(models), "tasks": len(next(iter(per.values()))), "samples_per_task": trials[models[0]].shape[1],
             "accuracy": {mo: float(trials[mo].mean()) for mo in models}, "pairs": len(pairs),
             "ratio_median_all": st.median(p["ratio"] for p in pairs), "close_pairs": len(close),
             "ratio_median_close": st.median(close), "ratio_range_close": [min(close), max(close)], "pair_rows": pairs}
        if e["samples_per_task"] > 1:
            # 每题多个样本:按样本算(把每个样本当独立)和按题算(先对每题取平均)的标准误之比
            e["task_vs_sample_se"] = []
            for mo in models:
                t = trials[mo]
                p = t.mean()
                if 0 < p < 1:
                    naive = math.sqrt(p * (1 - p) / (t.size - 1))
                    tse = t.mean(axis=1).std(ddof=1) / math.sqrt(t.shape[0])
                    e["task_vs_sample_se"].append({"model": mo, "accuracy": float(p), "sample_se": naive, "task_se": float(tse), "factor": float(tse / naive)})
        res["tasks"][task] = e
        print(f"{task:14s} {e['models']} models, {e['tasks']} tasks x {e['samples_per_task']}: paired/unpaired, pairs within "
              f"{CLOSE:.0%}: median {e['ratio_median_close']:.2f} [{min(close):.2f}, {max(close):.2f}] (n={len(close)}); all pairs {e['ratio_median_all']:.2f}")
        for c in e.get("task_vs_sample_se", []):
            print(f"    {c['model'][:36]:36s} acc {c['accuracy']:.2f}: task-level SE / per-sample SE = {c['factor']:.2f}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
