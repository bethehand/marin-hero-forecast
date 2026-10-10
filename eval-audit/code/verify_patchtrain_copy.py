"""Is shaochenze/PatchTrain's wikitext_document_level-test.json the whole WikiText-103 test set?

The Stack v3 scan found this file in stack-v3-train and scored it as written, with JSON escapes
(and StarPII masking) in the way. This decodes the GitHub copy and compares it with WikiText-103
raw test, which is identical to WikiText-2 raw test, the split lm-eval's `wikitext` task uses.

Usage: python -I verify_patchtrain_copy.py <wikitext_document_level-test.json> <wikitext-103-raw-v1 test parquet> <out.json>
"""

import hashlib
import json
import os
import re
import sys

import pyarrow.parquet as pq

N = 13


def grams(text):
    w = text.split()
    return {" ".join(w[i : i + N]) for i in range(len(w) - N + 1)}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    src, parquet, out = sys.argv[1:4]
    raw = open(src, encoding="utf-8").read()
    try:
        obj = json.loads(raw)
        docs = obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        docs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    decoded = "\n".join(d["text"] for d in docs)
    lines = pq.read_table(parquet).column("text").to_pylist()
    ev = "".join(lines)
    titles = [ln.strip() for ln in lines if re.fullmatch(r" = [^=].* = \n", ln)]
    g_ev = grams(ev)
    res = {
        "inputs": {"file": os.path.basename(src), "file_sha256": sha256(src), "eval_parquet_sha256": sha256(parquet)},
        "documents_in_file": len(docs),
        "eval_articles": len(titles),
        "eval_article_titles_in_file": sum(t in decoded for t in titles),
        "eval_13grams": len(g_ev),
        "share_as_written": round(len(g_ev & grams(raw)) / len(g_ev), 4),
        "share_after_json_decoding": round(len(g_ev & grams(decoded)) / len(g_ev), 4),
        "identical_after_whitespace_normalization": " ".join(decoded.split()) == " ".join(ev.split()),
    }
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
