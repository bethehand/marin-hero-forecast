#!/usr/bin/env python3
"""第 1 步:把 535B hero 预测要用的公开数据抓下来,存成带日期的快照。

数据来源(全部公开,只读,不往任何地方写):
  * W&B GraphQL(匿名):marin-community/marin_moe 里的阶梯 rung、hero 主线、团队的预测 run、
    以及 67B-A2B 回测要用的 run。
  * GitHub issue(通过 `gh` CLI 只读):#8435、#8506、#8824、#6044、#9126、#9874、#9615。
  * #8824 链接的 GCS 数据文件;mtracker 的 JSON 接口(页面文字由 LLM 生成,只取数字做交叉核对)。

用法:
  python fetch_snapshot.py --out ../data/snapshot_2026-10-09

输出目录结构:
  raw/wandb/<类别>/<run>/meta.json        run 的元数据(config、summary 全量)
  raw/wandb/<类别>/<run>/<指标组>.json     sampledHistory 原始行
  raw/github/issue_<n>.json               issue 正文 + 全部评论
  raw/external/*.json                     GCS 文件、mtracker 接口
  tables/wandb_long.csv                   长表:类别, run, step, 指标, 数值
  tables/runs.csv                         每个 run 一行的元数据
  manifest.json                           抓取时间、每个文件的大小和 sha256、错误记录
"""

import argparse
import concurrent.futures as cf
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://api.wandb.ai/graphql"
ENTITY, PROJECT = "marin-community", "marin_moe"

RUNS = {
    # 预注册阶梯(#8435);d2048 在约 81% 处崩溃
    "ladder": ["rav-ladder-d768-v2", "rav-ladder-d1024", "rav-ladder-d1536", "rav-ladder-d2048-v3"],
    # d768 上的配比对照。8/19 hero 开训用的是 epsilon0-monotonic-pi(PR #8452)
    "ladder_mix": [
        "rav-ladder-d768-v2-old-mix",
        "rav-ladder-d768-v2-epsilon0-monotonic-pi-57rows-candidate1-pergpu-prefetch32-w512-ts125g",
        "rav-ladder-d768-v2-epsilon0-pi-57rows-candidate1-pergpu-prefetch32-w512-ts125g",
        "rav-ladder-d768-v2-semantic-pi-56rows-candidate1-pergpu-prefetch32-w512-ts125g",
    ],
    # 10/3 重拟合(hero-dropless-refit-*)实际用的阶梯:H100 上的 d512–d1536,新旧两种配比
    # (名单取自该 run config 里的 source_runs)
    "ladder_h100": [
        "h100-mix25-20260912-d512",
        "h100-mix25-20260912-d768",
        "h100-mix25-20260912-d1024",
        "h100-mix25-20260912-d1536",
        "h100-ladder-d512-ep8-bs1024-791tpp-20260824-rno2a",
        "h100-ladder-d768-2xep8-bs1024-791tpp-20260824",
        "h100-ladder-d1024-ep8x4-bs1024-791tpp-20260825-rno2a-interactive",
        "h100-ladder-d1536-ep8x8-bs1024-791tpp-20260825-rno2a-interactive",
    ],
    # hero 主线,按交接顺序(#8506)
    "hero": [
        "hero-84579c80-072503",
        "hero-12d8b6f0-dee637",
        "hero-wd-gate-router-p02-step58k",
        "hero-ragged_a2a-nccl2307-ep-step81k",
        "hero-nopdl-step108k",
        "hero-main-step121638",
        "hero-fa4sm100-nomask-step146k",
    ],
    # 团队的预测 run
    "forecast": [
        "hero-forecast-seqlen4096-fc",
        "hero-forecast-swap-adjusted-20260914",
        "hero-forecast-uncheatable-swap-20260914",
        "hero-forecast-uncheatable-no-swap-20260914",
        "hero-forecast-uncheatable-asymptote-no-swap-20260914",
        "hero-forecast-uncheatable-asymptote-swap-20260914",
        "hero-forecast-uncheatable-asymptote-no-swap-from0-20260914",
        "hero-forecast-uncheatable-asymptote-swap-from0-20260914",
        "hero-uncheatable-ladder-floor14-no-swap-20260914",
        "hero-uncheatable-ladder-floor14-swap-20260914",
        "hero-uncheatable-joint-floor14-no-swap-20260914",
        "hero-uncheatable-joint-floor14-swap-20260914",
        "hero-uncheatable-joint-floor15-no-swap-20260914",
        "hero-uncheatable-joint-floor15-swap-20260914",
        "hero-uncheatable-monotone-floor15-old-5d24c2",
        "hero-uncheatable-monotone-floor15-swap-5d24c2",
        "hero-uncheatable-floor14-old-from30-75c36f",
        "hero-uncheatable-floor14-new-from30-75c36f",
        "hero-dropless-refit-old-20261003-phq0qbmf",
        "hero-dropless-refit-new-20261003-aa4pdwep",
    ],
    # 67B-A2B 回测(#6044):3 个锚点 + 2 段主训练
    "backtest_67b": [
        "june_prep_moe_may_d512_ep2_no_long_rope_seq8192_sw2k",
        "june_prep_moe_may_d768_ep2_no_long_rope_seq8192_sw2k",
        "june_prep_moe_may_d1024_ep2_bs512_no_long_rope_seq8192_sw2k",
        "moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T",
        "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
    ],
}

GITHUB_ISSUES = [8435, 8506, 8824, 6044, 9126, 9874, 9615]

EXTERNAL = {
    "gcs_8824_hero-bpb-forecast-analysis.json": "https://storage.googleapis.com/marin-public/held/hero-open-weight-ppl/2026.09.05.1/hero-bpb-forecast-analysis.json",
    "gcs_8824_paloma-subset-forecast-analysis.json": "https://storage.googleapis.com/marin-public/held/hero-open-weight-ppl/2026.09.05.1/paloma-subset-forecast-analysis.json",
    "gcs_8824_metrics.json": "https://storage.googleapis.com/marin-public/held/hero-open-weight-ppl/2026.09.05.1/metrics.json",
    "mtracker_hero-run-535b_loss.json": "https://mtracker.oa.dev/api/pages/hero-run-535b/loss",
}

EVAL_PREFIXES = ("eval_dropless/", "eval/", "fit/", "forecast/")
SKIP_SUFFIXES = ("loading_time", "total_time")
TRAIN_KEYS = ["optim/learning_rate", "train/loss", "train/cross_entropy_loss"]
REFERENCE_KEYS = ["eval_dropless/paloma/macro_loss", "eval/paloma/macro_loss"]
EVAL_SAMPLES = 100_000
TRAIN_SAMPLES = 4_000


def gql(query, variables=None, timeout=180, tries=5):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(ENDPOINT, data=body, headers={"Content-Type": "application/json"})
            out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            if out.get("errors"):
                raise RuntimeError(json.dumps(out["errors"])[:500])
            return out["data"]
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt + 1)
            print(f"    retry {attempt + 1} after: {str(e)[:120]}", flush=True)


def run_meta(name):
    q = """query($n: String!) { project(name: "%s", entityName: "%s") { run(name: $n) {
        name displayName createdAt heartbeatAt state config summaryMetrics } } }""" % (PROJECT, ENTITY)
    run = gql(q, {"n": name})["project"]["run"]
    if run is None:
        raise RuntimeError(f"run not found: {name}")
    run["config"] = json.loads(run["config"] or "{}")
    run["summaryMetrics"] = json.loads(run["summaryMetrics"] or "{}")
    return run


def sampled(name, keys, samples):
    q = """query($n: String!, $specs: [JSONString!]!) { project(name: "%s", entityName: "%s") {
        run(name: $n) { sampledHistory(specs: $specs) } } }""" % (PROJECT, ENTITY)
    spec = json.dumps({"keys": ["_step", *keys], "samples": samples})
    rows = gql(q, {"n": name, "specs": [spec]})["project"]["run"]["sampledHistory"][0]
    return sorted(rows, key=lambda r: r.get("_step", -1))


def key_groups(category, summary):
    keys = [k for k in summary if not k.startswith("_")]
    if category == "forecast":
        return {"all": sorted(k for k in keys if not k.endswith(SKIP_SUFFIXES))}
    groups = {}
    for k in keys:
        if not k.startswith(EVAL_PREFIXES) or k.endswith(SKIP_SUFFIXES):
            continue
        parts = k.split("/")
        g = "/".join(parts[:2]) if len(parts) > 2 else parts[0] + "/_top"
        groups.setdefault(g, []).append(k)
    return {g: sorted(v) for g, v in groups.items()}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def fetch_run(category, name, out):
    base = os.path.join(out, "raw", "wandb", category, name)
    log = []
    meta = run_meta(name)
    write_json(os.path.join(base, "meta.json"), meta)
    summary = meta["summaryMetrics"]
    groups = key_groups(category, summary)

    # 参照行数:只取一个最关键的指标,后面用来检查分组抓取有没有丢行
    ref_key = next((k for k in REFERENCE_KEYS if k in summary), None)
    ref_rows = sampled(name, [ref_key], EVAL_SAMPLES) if ref_key else []
    if ref_key:
        write_json(os.path.join(base, "reference_" + ref_key.replace("/", "__") + ".json"), ref_rows)

    n_rows = {}
    for g, keys in groups.items():
        rows = sampled(name, keys, EVAL_SAMPLES)
        # sampledHistory 只返回所有 key 都在的行;行数比参照少、或一行都没有时,逐个 key 重抓再按 step 合并
        if (ref_key and len(rows) < len(ref_rows)) or (not rows and keys):
            merged = {}
            for k in keys:
                for r in sampled(name, [k], EVAL_SAMPLES):
                    merged.setdefault(r["_step"], {"_step": r["_step"]}).update(r)
            log.append(f"{name}:{g} grouped={len(rows)} < ref={len(ref_rows)}; refetched per key -> {len(merged)}")
            rows = [merged[s] for s in sorted(merged)]
        n_rows[g] = len(rows)
        write_json(os.path.join(base, g.replace("/", "__") + ".json"), rows)

    if category in ("ladder", "ladder_mix", "ladder_h100", "hero", "backtest_67b"):
        tk = [k for k in TRAIN_KEYS if k in summary]
        if tk:
            rows = sampled(name, tk, TRAIN_SAMPLES)
            n_rows["train(sampled)"] = len(rows)
            write_json(os.path.join(base, "train_sampled.json"), rows)
    print(f"  [{category}] {name}: ref={len(ref_rows)} groups={n_rows}", flush=True)
    return {"category": category, "run": name, "meta": meta, "n_rows": n_rows, "ref_rows": len(ref_rows), "log": log}


def fetch_issue(n, out):
    issue = json.loads(subprocess.check_output(["gh", "api", f"repos/marin-community/marin/issues/{n}"], text=True))
    comments = json.loads(subprocess.check_output(
        ["gh", "api", "--paginate", "--slurp", f"repos/marin-community/marin/issues/{n}/comments?per_page=100"], text=True))
    comments = [c for page in comments for c in page]
    write_json(os.path.join(out, "raw", "github", f"issue_{n}.json"), {"issue": issue, "comments": comments})
    print(f"  [github] #{n}: {len(comments)} comments", flush=True)
    return n, len(comments)


def fetch_external(fname, url, out):
    path = os.path.join(out, "raw", "external", fname)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "hero-forecast"}), timeout=120) as r:
            data, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        data, status = e.read(), e.code
    with open(path, "wb") as f:
        f.write(data)
    print(f"  [external] {fname}: HTTP {status}, {len(data)} bytes", flush=True)
    return fname, status, len(data)


def to_float(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def cfg_value(cfg, key):
    v = cfg.get(key)
    return v.get("value") if isinstance(v, dict) and "value" in v else v


def build_tables(out, results):
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    with open(os.path.join(tdir, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "run", "display_name", "created_at", "heartbeat_at", "state",
                    "stop_after_steps", "num_train_steps", "summary_step", "ref_eval_rows"])
        for r in results:
            m = r["meta"]
            w.writerow([r["category"], r["run"], m["displayName"], m["createdAt"], m["heartbeatAt"], m["state"],
                        cfg_value(m["config"], "stop_after_steps"), cfg_value(m["config"], "num_train_steps"),
                        m["summaryMetrics"].get("_step"), r["ref_rows"]])
    n = 0
    with open(os.path.join(tdir, "wandb_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "run", "step", "key", "value"])
        for r in results:
            base = os.path.join(out, "raw", "wandb", r["category"], r["run"])
            seen = set()
            for fname in sorted(os.listdir(base)):
                if fname == "meta.json" or fname.startswith("reference_") or fname == "train_sampled.json":
                    continue
                for row in json.load(open(os.path.join(base, fname))):
                    step = row.get("_step")
                    for k, v in row.items():
                        fv = to_float(v)
                        if k.startswith("_") or fv is None or (step, k) in seen:
                            continue
                        seen.add((step, k))
                        w.writerow([r["category"], r["run"], step, k, repr(fv)])
                        n += 1
    with open(os.path.join(tdir, "train_sampled_long.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "run", "step", "key", "value"])
        for r in results:
            p = os.path.join(out, "raw", "wandb", r["category"], r["run"], "train_sampled.json")
            if os.path.exists(p):
                for row in json.load(open(p)):
                    for k, v in row.items():
                        fv = to_float(v)
                        if not k.startswith("_") and fv is not None:
                            w.writerow([r["category"], r["run"], row.get("_step"), k, repr(fv)])
    return n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out, "manifest.json")):
        sys.exit(f"refusing to overwrite an existing snapshot: {out}")
    os.makedirs(out, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    errors, results = [], []

    jobs = [(c, n) for c, names in RUNS.items() for n in names]
    print(f"W&B: {len(jobs)} runs", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_run, c, n, out): (c, n) for c, n in jobs}
        for fut in cf.as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                errors.append(f"wandb {futs[fut]}: {e}")
                print(f"  ERROR {futs[fut]}: {e}", flush=True)
    order = {n: i for i, (_, n) in enumerate(jobs)}
    results.sort(key=lambda r: order[r["run"]])

    print("GitHub issues:", flush=True)
    issues = {}
    for n in GITHUB_ISSUES:
        try:
            issues[n] = fetch_issue(n, out)[1]
        except Exception as e:
            errors.append(f"github #{n}: {e}")
    print("External files:", flush=True)
    external = {}
    for fname, url in EXTERNAL.items():
        try:
            external[fname] = {"url": url, "status": fetch_external(fname, url, out)[1]}
        except Exception as e:
            errors.append(f"external {fname}: {e}")

    n_long = build_tables(out, results)
    files = []
    for root, _, names in os.walk(out):
        for fname in sorted(names):
            p = os.path.join(root, fname)
            files.append({"path": os.path.relpath(p, out), "bytes": os.path.getsize(p), "sha256": sha256(p)})
    script = os.path.abspath(__file__)
    manifest = {
        "started_utc": started,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": os.path.relpath(script, out),
        "script_sha256": sha256(script),
        "wandb": {"endpoint": ENDPOINT, "project": f"{ENTITY}/{PROJECT}", "eval_samples": EVAL_SAMPLES,
                  "train_samples": TRAIN_SAMPLES,
                  "runs": [{"category": r["category"], "run": r["run"], "rows": r["n_rows"],
                            "ref_rows": r["ref_rows"], "log": r["log"]} for r in results]},
        "github_issues": issues,
        "external": external,
        "wandb_long_rows": n_long,
        "errors": errors,
        "files": sorted(files, key=lambda x: x["path"]),
    }
    write_json(os.path.join(out, "manifest.json"), manifest)
    print(f"done: {len(results)} runs, {n_long} long rows, {len(errors)} errors -> {out}", flush=True)


if __name__ == "__main__":
    main()
