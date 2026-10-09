#!/bin/sh
# Fetch the third-party inputs of decon_recall_windowed.py at pinned commits.
# They are not redistributed here. Each source goes into its own new directory.
set -eu
out=${1:-../inputs}
nplm=5aeb36125a26e56acaac4843525e70c44cd3a2a4   # quanpn90/torch-nplm, 2016-01-07
lstm=76870253cfca069477f06b7056af87f98490b6eb   # wojzaremba/lstm, 2015-05-20 (independent PTB copy)
marin=3c62e65f8f56c2f4bb6f290a8e27dbc294bec848  # marin-community/marin main, 2026-10-07
mkdir -p "$out/torch-nplm" "$out/wojzaremba-lstm"
for f in README.md extract_ngrams.py train.lua data/ptb/test.6grams data/ptb/valid.6grams data/ptb/valid.txt \
         data/ptb/test.txt util/BatchLoader.lua util/IdentityLinear.lua; do
  mkdir -p "$out/torch-nplm/$(dirname "$f")"
  curl -sSfL "https://raw.githubusercontent.com/quanpn90/torch-nplm/$nplm/$f" -o "$out/torch-nplm/$f"
done
curl -sSfL "https://raw.githubusercontent.com/wojzaremba/lstm/$lstm/data/ptb.valid.txt" -o "$out/wojzaremba-lstm/ptb.valid.txt"
git clone -q https://github.com/marin-community/marin "$out/marin"
git -C "$out/marin" checkout -q "$marin"
shasum -a 256 "$out/torch-nplm/data/ptb/valid.txt" "$out/wojzaremba-lstm/ptb.valid.txt" \
  "$out/marin/lib/marin/src/marin/datakit/decon.py"
