"""Can Marin's decontamination matcher catch the torch-nplm copy of PTB?

Runs Marin's own matching code (lib/marin/src/marin/datakit/decon.py, loaded
from a checkout without installing its dependencies) with the production
settings from experiments/datakit/reference_pipeline.py: 13-grams, overlap
threshold 0.5, at least 2 matched features, paragraphs split on "\n\n".

The eval side is the PTB validation set (Paloma `ptb` val is this text). It is
indexed as if Paloma were in the decon eval corpus, which today it is not, so
every result here is a best case for the current matcher.

The training side is the quanpn90/torch-nplm repository rendered the way
Marin's Stack v3 downloader renders a repository (row_to_doc in
lib/marin/src/marin/datakit/download/stack_v3.py), with the files #9874 lists
for the trained document.

It also tests a candidate fix, "de-windowing": when most lines of a paragraph
are sliding windows that shift by one token, stitch them back into running
text, drop the padding tokens, and score the stitched text as one more
paragraph.

Needs Python 3.10+ (NGramConfig uses dataclass kw_only). Usage:
    python -I decon_recall_windowed.py <marin checkout> <torch-nplm dir> <ptb.valid.txt> <out dir> [negative-control dirs...]
"""

import ast
import collections
import dataclasses
import hashlib
import itertools
import json
import os
import re
import sys
import time
import types
import typing
from pathlib import Path, PurePosixPath

DECON_REL = "lib/marin/src/marin/datakit/decon.py"
PROD = dict(ngram_length=13, stride=0, overlap_threshold=0.5, min_matched_features=2, paragraph_delimiter="\n\n")
TRAINED_FILES = [  # the files #9874 lists for the trained document
    "README.md", "extract_ngrams.py", "train.lua", "data/ptb/test.6grams", "data/ptb/valid.6grams",
    "util/BatchLoader.lua", "util/IdentityLinear.lua",
]
PAD = {"<s>", "</s>"}
MIN_WINDOW, MIN_LINES, MIN_SHIFT_FRAC = 3, 8, 0.5

NEEDED = {
    "_blake2b", "MIN_SHORT_EXACT_TOKENS", "DEFAULT_PARAGRAPH_DELIMITER", "LARGE_TEXT_STREAMING_THRESHOLD",
    "_TOKEN_PATTERN", "NGramConfig", "_bloom_hash", "_has_alpha", "_extract_token_ngrams",
    "_extract_streaming_ngrams", "_extract_ngrams", "_short_exact_feature_from_tokens", "_short_exact_feature",
    "_iter_nonempty_segments", "_iter_paragraphs", "_extract_features", "_ParagraphOverlap",
    "_paragraph_overlap_matches_and_presence", "_document_overlap_and_matches",
}


def load_decon(checkout):
    """Exec only the pure matching functions of Marin's decon.py, unmodified."""
    path = Path(checkout) / DECON_REL
    src = path.read_text()
    keep = []
    for node in ast.parse(src).body:
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & NEEDED:
            keep.append(ast.get_source_segment(src, node, padded=False))
    mod = types.ModuleType("decon_subset")  # dataclass() looks its module up in sys.modules
    sys.modules[mod.__name__] = mod
    ns = mod.__dict__
    ns.update(
        hashlib=hashlib, re=re, deque=collections.deque, islice=itertools.islice, chain=itertools.chain,
        Iterator=typing.Iterator, Container=typing.Container, NamedTuple=typing.NamedTuple,
        dataclass=dataclasses.dataclass,
    )
    # get_source_segment drops decorators; NGramConfig needs @dataclass(frozen=True, kw_only=True).
    code = "\n\n".join(keep).replace("class NGramConfig:", "@dataclass(frozen=True, kw_only=True)\nclass NGramConfig:")
    exec(compile(code, str(path), "exec"), ns)
    missing = NEEDED - set(ns)
    assert not missing, missing
    return ns, hashlib.sha256(src.encode()).hexdigest()


def render_repo(repo_path, root, files):
    """Same layout as stack_v3.row_to_doc: directory-block DFS, "\n\n" between sections."""
    by_dir, children = {}, {}
    for f in files:
        d = PurePosixPath(f).parts[:-1]
        by_dir.setdefault(d, []).append(f)
        for k in range(len(d)):
            children.setdefault(d[:k], set()).add(d[: k + 1])
    ordered, pending = [], [()]
    while pending:
        d = pending.pop()
        ordered.extend(sorted(by_dir.get(d, []), key=lambda f: PurePosixPath(f).name))
        pending.extend(sorted(children.get(d, ()), reverse=True))
    sections = [f"Repository: {repo_path}"]
    sections += [f"File: {f}\n{(Path(root) / f).read_text()}" for f in ordered]
    return "\n\n".join(sections), ordered


def dewindow(paragraph):
    """Stitch shift-by-one sliding windows back into running text, or return None.

    Consecutive lines a, b with the same width w >= MIN_WINDOW continue one run
    when b[:-1] == a[1:]. The paragraph qualifies when at least MIN_SHIFT_FRAC of
    its lines (and at least MIN_LINES) continue a run. Padding tokens are dropped.
    """
    lines = [ln.split() for ln in paragraph.split("\n")]
    lines = [t for t in lines if t]
    if len(lines) < MIN_LINES:
        return None
    out, shifted, prev = [], 0, None
    for t in lines:
        if prev is not None and len(t) == len(prev) >= MIN_WINDOW and t[:-1] == prev[1:]:
            out.append(t[-1])
            shifted += 1
        else:
            out.extend(t)
        prev = t
    if shifted < MIN_LINES or shifted < MIN_SHIFT_FRAC * len(lines):
        return None
    return " ".join(w for w in out if w not in PAD)


def score(ns, cfg, text, bloom, with_dewindow=False):
    max_score, matched = ns["_document_overlap_and_matches"](text, bloom, cfg)
    res = {"max_overlap": round(max_score, 4), "matched_features": len(matched)}
    if with_dewindow:
        extra = [s for s in (dewindow(p) for p in text.split(cfg.paragraph_delimiter)) if s]
        res["dewindowed_paragraphs"] = len(extra)
        for s in extra:
            o = ns["_paragraph_overlap_matches_and_presence"](s, bloom, cfg)
            if o.score > max_score:
                max_score = o.score
            if o.score >= cfg.overlap_threshold and len(set(o.matched_hashes)) >= cfg.min_matched_features:
                matched = sorted(set(matched) | set(o.matched_hashes))
        res.update(max_overlap=round(max_score, 4), matched_features=len(matched))
    res["flagged"] = bool(matched)
    return res


def paragraph_table(ns, cfg, text, bloom):
    rows = []
    for p in text.split(cfg.paragraph_delimiter):
        o = ns["_paragraph_overlap_matches_and_presence"](p, bloom, cfg)
        head = p.split("\n", 1)[0][:60]
        rows.append({"paragraph": head, "chars": len(p), "features": o.feature_count,
                     "matched": len(o.matched_hashes), "overlap": round(o.score, 4)})
    return rows


def windows(text, width):
    """Sliding windows as torch-nplm's extract_ngrams.py writes them: one per word, left-padded with <s>."""
    out = []
    for line in text.split("\n"):
        sent = ["<s>"] + line.split()
        for i in range(1, len(sent)):
            w = sent[max(0, i + 1 - width) : i + 1]
            out.append(" ".join(["<s>"] * (width - len(w)) + w))
    return "\n".join(out) + "\n"


def best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def negative_control(dirs, exts=(".py", ".md", ".txt", ".lua", ".json", ".yaml", ".rst", ".csv", ".c", ".h")):
    n_files = n_par = n_hit = 0
    hits = []
    for d in dirs:
        for p in Path(d).rglob("*"):
            if "site-packages" in p.parts or not p.is_file() or p.suffix not in exts or p.stat().st_size > 5_000_000:
                continue
            try:
                txt = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            n_files += 1
            for para in txt.split("\n\n"):
                n_par += 1
                if dewindow(para):
                    n_hit += 1
                    hits.append(str(p))
    return {"files": n_files, "paragraphs": n_par, "dewindow_triggered": n_hit, "examples": hits[:10]}


def main():
    checkout, repo_dir, ptb_valid, out_dir, *neg_dirs = sys.argv[1:]
    ns, decon_sha = load_decon(checkout)
    cfg = ns["NGramConfig"](**PROD)
    os.makedirs(out_dir, exist_ok=True)

    eval_text = Path(ptb_valid).read_text()
    bloom = {ns["_bloom_hash"](f) for f in ns["_extract_features"](eval_text, cfg)}

    trained, order = render_repo("quanpn90/torch-nplm", repo_dir, TRAINED_FILES)
    with_raw, _ = render_repo("quanpn90/torch-nplm", repo_dir, TRAINED_FILES + ["data/ptb/valid.txt"])
    raw_only, _ = render_repo("someone/ptb-copy", repo_dir, ["data/ptb/valid.txt"])
    windows_only, _ = render_repo("someone/ptb-windows", repo_dir, ["data/ptb/valid.6grams"])

    cases = {
        "torch-nplm as trained (6-gram windows)": score(ns, cfg, trained, bloom),
        "torch-nplm + raw data/ptb/valid.txt": score(ns, cfg, with_raw, bloom),
        "raw valid.txt alone": score(ns, cfg, raw_only, bloom),
        "valid.6grams alone": score(ns, cfg, windows_only, bloom),
    }
    # #9874: the trained copy has "marvin davis" scrubbed to <NAME> inside windows.
    scrubbed_6g = re.sub(r"\bmarvin davis\b", "<NAME>", Path(repo_dir, "data/ptb/valid.6grams").read_text())
    scrubbed = f"Repository: someone/ptb-windows\n\nFile: data/ptb/valid.6grams\n{scrubbed_6g}"
    fixed = {
        "torch-nplm as trained, with de-windowing": score(ns, cfg, trained, bloom, with_dewindow=True),
        "valid.6grams alone, with de-windowing": score(ns, cfg, windows_only, bloom, with_dewindow=True),
        "valid.6grams with marvin davis -> <NAME>, with de-windowing": score(ns, cfg, scrubbed, bloom, with_dewindow=True),
        "raw valid.txt alone, with de-windowing": score(ns, cfg, raw_only, bloom, with_dewindow=True),
    }
    sweep = {}
    for width in range(3, 31):
        doc = f"Repository: someone/windows\n\nFile: data/ptb/valid.{width}grams\n{windows(eval_text, width)}"
        cur, dew = score(ns, cfg, doc, bloom), score(ns, cfg, doc, bloom, with_dewindow=True)
        sweep[width] = {"current_overlap": cur["max_overlap"], "current_flagged": cur["flagged"],
                        "dewindow_overlap": dew["max_overlap"], "dewindow_flagged": dew["flagged"]}
    assert windows(eval_text, 6) == Path(repo_dir, "data/ptb/valid.6grams").read_text()

    paragraphs = trained.split(cfg.paragraph_delimiter)
    t_match = best_of(lambda: ns["_document_overlap_and_matches"](trained, bloom, cfg))
    t_dewin = best_of(lambda: [dewindow(p) for p in paragraphs])

    stitched = dewindow(Path(repo_dir, "data/ptb/valid.6grams").read_text())
    raw_tokens = eval_text.split()
    stitched_tokens = stitched.split()
    result = {
        "decon_py_sha256": decon_sha,
        "settings": PROD,
        "eval_ptb_valid_sha256": hashlib.sha256(eval_text.encode()).hexdigest(),
        "eval_features_in_bloom": len(bloom),
        "trained_doc_chars": len(trained),
        "trained_doc_file_order": order,
        "current_matcher": cases,
        "with_dewindowing": fixed,
        "window_width_sweep": sweep,
        "stitched_equals_eval_tokens": stitched_tokens == raw_tokens,
        "stitched_tokens": len(stitched_tokens),
        "eval_tokens": len(raw_tokens),
        "seconds_on_trained_doc": {"current_matcher": round(t_match, 3), "dewindow_pass": round(t_dewin, 3)},
        "paragraphs_trained_doc_ge_100_features": [
            r for r in paragraph_table(ns, cfg, trained, bloom) if r["features"] >= 100
        ],
    }
    if neg_dirs:
        result["negative_control"] = negative_control(neg_dirs)
    Path(out_dir, "decon_recall_windowed.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
