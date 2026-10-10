#!/bin/sh
# Fetch the third-party inputs of decon_recall_windowed.py, verify_patchtrain_copy.py and
# check_paloma_wikitext_split.py at pinned commits.
# They are not redistributed here. Each source goes into its own new directory.
set -eu
out=${1:-../inputs}
nplm=5aeb36125a26e56acaac4843525e70c44cd3a2a4   # quanpn90/torch-nplm, 2016-01-07
lstm=76870253cfca069477f06b7056af87f98490b6eb   # wojzaremba/lstm, 2015-05-20 (independent PTB copy)
marin=3c62e65f8f56c2f4bb6f290a8e27dbc294bec848  # marin-community/marin main, 2026-10-07
patch=eda31cff397e63835877c86550845ceca0fb378e  # shaochenze/PatchTrain main, 2025-11-10
pval=77b22696d8d4aec50280e2a506e6522770d1e171   # mlfoundations/paloma_validation (ungated Paloma token mirror)
neox=c292233c833e336628618a88a648727eb3dff0a7   # EleutherAI/gpt-neox-20b (tokenizer)
mkdir -p "$out/torch-nplm" "$out/wojzaremba-lstm" "$out/patchtrain" "$out/wikitext-103-raw-v1-test" \
  "$out/wikitext-103-raw-v1-validation" "$out/paloma-wikitext-val" "$out/gpt-neox-20b"
for f in README.md extract_ngrams.py train.lua data/ptb/test.6grams data/ptb/valid.6grams data/ptb/valid.txt \
         data/ptb/test.txt util/BatchLoader.lua util/IdentityLinear.lua; do
  mkdir -p "$out/torch-nplm/$(dirname "$f")"
  curl -sSfL "https://raw.githubusercontent.com/quanpn90/torch-nplm/$nplm/$f" -o "$out/torch-nplm/$f"
done
curl -sSfL "https://raw.githubusercontent.com/wojzaremba/lstm/$lstm/data/ptb.valid.txt" -o "$out/wojzaremba-lstm/ptb.valid.txt"
curl -sSfL "https://raw.githubusercontent.com/shaochenze/PatchTrain/$patch/wikitext_document_level-test.json" \
  -o "$out/patchtrain/wikitext_document_level-test.json"
curl -sSfL "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-103-raw-v1/test/0.parquet" \
  -o "$out/wikitext-103-raw-v1-test/0.parquet"
curl -sSfL "https://huggingface.co/api/datasets/Salesforce/wikitext/parquet/wikitext-103-raw-v1/validation/0.parquet" \
  -o "$out/wikitext-103-raw-v1-validation/0.parquet"
curl -sSfL "https://huggingface.co/datasets/mlfoundations/paloma_validation/resolve/$pval/coarse_domains/wikitext_103/val/00000001.tar" \
  -o "$out/paloma-wikitext-val/val.tar"
curl -sSfL "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/$neox/tokenizer.json" -o "$out/gpt-neox-20b/tokenizer.json"
git clone -q https://github.com/marin-community/marin "$out/marin"
git -C "$out/marin" checkout -q "$marin"
shasum -a 256 "$out/torch-nplm/data/ptb/valid.txt" "$out/wojzaremba-lstm/ptb.valid.txt" \
  "$out/marin/lib/marin/src/marin/datakit/decon.py" "$out/patchtrain/wikitext_document_level-test.json" \
  "$out/wikitext-103-raw-v1-test/0.parquet" "$out/wikitext-103-raw-v1-validation/0.parquet" \
  "$out/paloma-wikitext-val/val.tar"
