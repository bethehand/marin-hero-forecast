"""Find copies of the PTB and WikiText-103 eval texts in The Stack v3 (stack-v3-train).

Marin's hero trains on Stack v3 repositories rendered as one document each. Marin pinned
revision bb2fa950 (v3.0), whose files can no longer be downloaded (history squashed on
2026-09-25). This scans the current revision (v3.1): the same pipeline output minus opted-out
repositories and minus the repeated file entries that Marin already drops itself (marin#7788).

Stage A reads only two small adjacent columns (file_path, size_bytes; about 46 GB of the
3.7 TB dataset, one HTTP range per row group) and keeps a file when either
  - its path looks like PTB or WikiText data (a path component named ptb / penn / penntreebank /
    treebank / simple-examples, "wikitext" anywhere, or names like wiki.valid.tokens), it is at
    least 50 KB, and it is not a source-code file; or
  - its name says valid/test/val/dev, it is plain text (.txt/.tokens/.raw or no extension), and
    its size is within 6% of a PTB or WikiText valid/test file (PII redaction changes sizes a bit).
Stage B downloads the content of the row groups that hold those files and counts 13-gram
matches against PTB valid/test and WikiText-103 valid/test (tokens and raw), on the file as
written and after "de-windowing" (stitching shift-by-one sliding windows back into text).

Runs anywhere with fast access to Hugging Face (on a Google Colab CPU runtime, the full scan took 3 hours):
  pip install pyarrow requests
  python stackv3_eval_copy_scan.py --out results            # full scan
  python stackv3_eval_copy_scan.py --out test --max-files 2 # quick test
No Hugging Face token is needed; the dataset is public and ungated.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

REPO = "HuggingFaceCode/stack-v3-train"
REV = "1f61b735bc0a5698345ce2196730f24bfa467f33"  # v3.1, 2026-09-25
HF = "https://huggingface.co"
PATH_RE = (r"(^|[/_.\-])(ptb|penn|penntreebank|treebank|simple-examples)([/_.\-]|$)"
           r"|wikitext|(^|/)wiki\.(train|valid|test)\.(tokens|raw)$")
CODE_EXT_RE = (r"\.(py|pyc|js|ts|jsx|tsx|java|c|cc|cpp|cxx|h|hpp|go|rs|rb|php|cs|scala|kt|swift|m|mm|lua|sh|"
               r"r|jl|ipynb|html|htm|css|xml|yml|yaml|toml|cfg|ini|md|rst|tex|bib|sql|pl|pm|hs|ml|ex|exs|erl|"
               r"clj|dart|vue|svelte|gradle|cmake|mk|bat|ps1|lock|svg)$")
NAME_RE = r"(valid|test|val|dev)"
TEXT_EXT_RE = r"(\.(txt|tokens|raw)|^[^.]+)$"
# PTB valid/test (Mikolov), WikiText-2/103 valid/test, tokens and raw (bytes)
CANON_SIZES = (399_782, 449_945, 1_119_083, 1_139_711, 1_144_248, 1_253_515, 1_278_143, 1_287_656)
MIN_PATH_BYTES = 50_000
EVALS = {  # name -> source of the eval text
    "ptb_valid": "https://raw.githubusercontent.com/wojzaremba/lstm/76870253cfca069477f06b7056af87f98490b6eb/data/ptb.valid.txt",
    "ptb_test": "https://raw.githubusercontent.com/wojzaremba/lstm/76870253cfca069477f06b7056af87f98490b6eb/data/ptb.test.txt",
    "wt103_valid": "wikitext-103-v1/validation",
    "wt103_test": "wikitext-103-v1/test",
    "wt103raw_valid": "wikitext-103-raw-v1/validation",
    "wt103raw_test": "wikitext-103-raw-v1/test",
}
N = 13
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def get(url, **kw):
    for attempt in range(8):
        try:
            r = session().get(url, timeout=120, **kw)
            if r.status_code in (200, 206, 302, 307):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(min(60, 2 ** attempt))
    raise IOError(f"GET failed: {url[:120]}")


def list_files():
    out, url = [], f"{HF}/api/datasets/{REPO}/tree/{REV}/data"
    while url:
        r = get(url)
        out += [(e["path"], e["size"]) for e in r.json() if e["type"] == "file" and e["path"].endswith(".parquet")]
        url = r.links.get("next", {}).get("url")
    return sorted(out)


def cdn_url(path):
    url = f"{HF}/datasets/{REPO}/resolve/{REV}/{path}"
    for _ in range(5):
        r = get(url, allow_redirects=False)
        if r.status_code in (302, 307) and "location" in r.headers:
            loc = r.headers["location"]
            url = loc if loc.startswith("http") else HF + loc
            if "huggingface.co" not in url.split("/")[2]:
                return url
        else:
            return url
    return url


class RangeFile(io.RawIOBase):
    """Read-only file over HTTP range requests (pyarrow reads only the byte ranges it needs)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        data = get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_parquet(path, size):
    return pq.ParquetFile(RangeFile(cdn_url(path), size), pre_buffer=True)


def candidate_mask(fp, sz):
    """Boolean mask over files: PTB/WikiText-looking paths, or eval-sized plain-text valid/test files."""
    low = pc.fill_null(pc.utf8_lower(fp), "")
    base = pc.replace_substring_regex(low, r".*/", "")
    size = pc.fill_null(sz, 0)
    rule_path = pc.and_(pc.and_(pc.match_substring_regex(low, PATH_RE), pc.greater_equal(size, MIN_PATH_BYTES)),
                        pc.invert(pc.match_substring_regex(base, CODE_EXT_RE)))
    size_ok = None
    for c in CANON_SIZES:
        m = pc.and_(pc.greater_equal(size, int(c * 0.94)), pc.less_equal(size, int(c * 1.06)))
        size_ok = m if size_ok is None else pc.or_(size_ok, m)
    rule_size = pc.and_(pc.and_(pc.match_substring_regex(base, NAME_RE), pc.match_substring_regex(base, TEXT_EXT_RE)),
                        size_ok)
    return pc.fill_null(pc.or_(rule_path, rule_size), False)


def stage_a(path, size):
    """Candidate files in one parquet file. Reads two adjacent columns: one HTTP range per row group."""
    pf = open_parquet(path, size)
    rows = []
    for g in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(g, columns=["files.list.element.file_path", "files.list.element.size_bytes"])
        files = t.column("files").combine_chunks()
        flat = pc.list_flatten(files)
        fp = flat.field("file_path")
        idx = pc.indices_nonzero(candidate_mask(fp, flat.field("size_bytes")))
        if len(idx) == 0:
            continue
        for r, p, s in zip(pc.take(pc.list_parent_indices(files), idx).to_pylist(), pc.take(fp, idx).to_pylist(),
                           pc.take(flat.field("size_bytes"), idx).to_pylist()):
            rows.append({"parquet": path, "row_group": g, "row": r, "file_path": p, "size_bytes": s})
    return pf.metadata.num_rows, rows


def ngrams(tokens):
    return {" ".join(tokens[i : i + N]) for i in range(len(tokens) - N + 1)}


def dewindow(text, min_lines=8, min_frac=0.5):
    lines = [ln.split() for ln in text.split("\n")]
    lines = [t for t in lines if t]
    if len(lines) < min_lines:
        return None
    out, shifted, prev = [], 0, None
    for t in lines:
        if prev is not None and len(t) == len(prev) >= 3 and t[:-1] == prev[1:]:
            out.append(t[-1])
            shifted += 1
        else:
            out.extend(t)
        prev = t
    if shifted < min_lines or shifted < min_frac * len(lines):
        return None
    return " ".join(w for w in out if w not in ("<s>", "</s>"))


def load_evals():
    sets = {}
    for name, src in EVALS.items():
        if src.startswith("http"):
            text = get(src).text
        else:
            cfg, split = src.split("/")
            url = f"{HF}/api/datasets/Salesforce/wikitext/parquet/{cfg}/{split}/0.parquet"
            table = pq.read_table(io.BytesIO(get(url).content))
            text = "".join(table.column("text").to_pylist())
        sets[name] = ngrams(text.split())
        print(f"eval {name}: {len(sets[name]):,} distinct {N}-grams", flush=True)
    return sets


def score(text, sets):
    best = {}
    for label, t in (("as_written", text), ("dewindowed", dewindow(text))):
        if not t:
            continue
        grams = ngrams(t.split())
        for name, s in sets.items():
            hits = len(grams & s)
            frac_eval = hits / max(1, len(s))  # share of the eval's 13-grams present in this file
            key = f"{name}"
            if hits > best.get(key, (0, 0, ""))[0]:
                best[key] = (hits, frac_eval, label)
    return best


def stage_b(group, sizes, sets):
    path, g = group["parquet"], group["row_group"]
    pf = open_parquet(path, sizes[path])
    t = pf.read_row_group(g, columns=["repo_path", "github_metadata.stars", "files.list.element.file_path",
                                      "files.list.element.content"])
    rows = sorted({c["row"] for c in group["cands"]})
    t = t.take(pa.array(rows))  # only the candidate repos become Python objects
    want = {}
    for c in group["cands"]:
        want.setdefault(c["row"], set()).add(c["file_path"])
    out = []
    for r, repo, meta, files in zip(rows, t.column("repo_path").to_pylist(), t.column("github_metadata").to_pylist(),
                                    t.column("files").to_pylist()):
        seen = set()
        for f in files or []:
            if f["file_path"] not in want[r] or f["file_path"] in seen:
                continue
            seen.add(f["file_path"])
            text = f["content"] or ""
            row = {"parquet": path, "row_group": g, "row": r, "repo_path": repo, "stars": (meta or {}).get("stars"),
                   "file_path": f["file_path"], "chars": len(text)}
            best = score(text, sets)
            for name in EVALS:
                hits, frac, label = best.get(name, (0, 0.0, ""))
                row[f"{name}_hits"] = hits
                row[f"{name}_eval_share"] = round(frac, 4)
                row[f"{name}_layout"] = label
            out.append(row)
    return out


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stackv3_scan")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32, help="parallel files in stage A")
    ap.add_argument("--b-workers", type=int, default=8, help="parallel row groups in stage B (memory-bound)")
    ap.add_argument("--min-hits", type=int, default=100)
    ap.add_argument("--path-re", default=None, help="override the path pattern (for testing)")
    args = ap.parse_args()
    global PATH_RE
    if args.path_re:
        PATH_RE = args.path_re
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    files = list_files()
    print(f"{len(files)} parquet files at {REV[:10]}", flush=True)
    if args.max_files:
        files = files[: args.max_files]
    sizes = dict(files)

    # Stage A
    cands, repos, done, failed = [], 0, 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(stage_a, p, s): p for p, s in files}
        for fut in as_completed(futs):
            try:
                n, rows = fut.result()
                repos += n
                cands += rows
            except Exception as exc:  # keep going; retried below
                failed.append(futs[fut])
                print(f"stage A failed on {futs[fut]}: {str(exc)[:150]}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(files):
                print(f"stage A {done}/{len(files)} files, {repos:,} repos, {len(cands):,} candidate files, "
                      f"{len(failed)} failed, {time.time() - t0:.0f}s", flush=True)
    still_failed = []
    for p in failed:  # one sequential retry
        try:
            n, rows = stage_a(p, sizes[p])
            repos += n
            cands += rows
        except Exception as exc:
            still_failed.append(p)
            print(f"stage A retry failed on {p}: {str(exc)[:150]}", flush=True)
    write_csv(os.path.join(args.out, "stage_a_candidates.csv"), cands)

    # Stage B: content of the row groups that hold candidates
    sets = load_evals()
    groups = {}
    for c in cands:
        groups.setdefault((c["parquet"], c["row_group"]), {"parquet": c["parquet"], "row_group": c["row_group"],
                                                          "cands": []})["cands"].append(c)
    print(f"stage B: {len(cands):,} candidate files in {len(groups):,} row groups", flush=True)
    scored, done, b_failed = [], 0, []
    with ThreadPoolExecutor(args.b_workers) as ex:
        futs = {ex.submit(stage_b, g, sizes, sets): k for k, g in groups.items()}
        for fut in as_completed(futs):
            try:
                scored += fut.result()
            except Exception as exc:
                b_failed.append(futs[fut])
                print(f"stage B failed on {futs[fut]}: {str(exc)[:150]}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"stage B {done}/{len(futs)} row groups, {time.time() - t0:.0f}s", flush=True)
    hits_cols = [f"{n}_hits" for n in EVALS]
    scored.sort(key=lambda r: -max(r[c] for c in hits_cols))
    write_csv(os.path.join(args.out, "stage_b_scored.csv"), scored)
    matches = [r for r in scored if max(r[c] for c in hits_cols) >= args.min_hits]
    write_csv(os.path.join(args.out, "matches.csv"), matches)
    summary = {"revision": REV, "parquet_files": len(files), "repos": repos, "stage_a_candidates": len(cands),
               "stage_b_row_groups": len(groups), "matches_min_hits": args.min_hits, "matches": len(matches),
               "stage_a_failed_files": still_failed, "stage_b_failed_row_groups": [list(k) for k in b_failed],
               "seconds": round(time.time() - t0)}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in matches[:40]:
        top = max(EVALS, key=lambda n: r[f"{n}_hits"])
        print(f"{r['repo_path']} :: {r['file_path']} stars={r['stars']} best={top} hits={r[f'{top}_hits']} "
              f"share={r[f'{top}_eval_share']} ({r[f'{top}_layout']})")


if __name__ == "__main__":
    main()
