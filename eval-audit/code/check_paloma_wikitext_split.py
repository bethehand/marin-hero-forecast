"""Which WikiText-103 split is Paloma's `wikitext_103` val?

allenai/paloma is gated, so this uses the ungated token mirror mlfoundations/paloma_validation
(GPT-NeoX-20B token sequences of 2,049 tokens). It decodes the val shard and counts how many
13-grams it shares with WikiText-103 raw validation and test.

Usage: python -I check_paloma_wikitext_split.py <val.tar> <gpt-neox-20b tokenizer.json> <validation parquet> <test parquet> <out.json>
"""

import gzip
import hashlib
import json
import sys
import tarfile

import pyarrow.parquet as pq
from tokenizers import Tokenizer

N = 13


def grams(text):
    w = text.split()
    return {" ".join(w[i : i + N]) for i in range(len(w) - N + 1)}


def main():
    tar, tok, valid_pq, test_pq, out = sys.argv[1:6]
    tokenizer = Tokenizer.from_file(tok)
    with tarfile.open(tar) as tf:
        seqs = [json.loads(gzip.decompress(tf.extractfile(m).read())) for m in tf.getmembers() if m.isfile()]
    paloma = grams("\n".join(tokenizer.decode(s) for s in seqs))
    res = {"val_tar_sha256": hashlib.sha256(open(tar, "rb").read()).hexdigest(), "sequences": len(seqs),
           "paloma_val_13grams": len(paloma)}
    for name, path in (("validation", valid_pq), ("test", test_pq)):
        ev = grams("".join(pq.read_table(path).column("text").to_pylist()))
        res[f"wt103_{name}_share_in_paloma_val"] = round(len(ev & paloma) / len(ev), 4)
        res[f"paloma_val_share_in_wt103_{name}"] = round(len(ev & paloma) / len(paloma), 4)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
