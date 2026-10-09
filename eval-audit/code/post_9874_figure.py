#!/usr/bin/env python3
"""Numbers and figure for the #9874 reply, from the frozen W&B snapshot.

Prints the eval values the reply quotes and writes results/post/fig_9874.png:
PTB and WikiText-103 (Paloma) on the hero, with the torch-nplm and c28q1 reads
#9874 mapped, and the windows we suggest scanning. m2d2_wikipedia_unsplit is
the Wikipedia control for WikiText.

Usage: python post_9874_figure.py <snapshot dir> <out dir>
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_eval_events import load  # noqa: E402  same loader and run lineage as phase 1

READS = {  # #9874, ClassicLarry 2026-10-07
    "torch-nplm, PTB test split only": (176_080, 176_355),
    "c28q1 n-gram tables": (206_300, 206_300),
    "torch-nplm, all of PTB valid": (209_390, 209_669),
}
SCAN = {"ptb": [(102_000, 104_999), (189_000, 191_999)], "wikitext_103": [(126_000, 128_999)]}
STEPS = {
    "ptb": [101_999, 104_999, 107_999, 173_999, 176_999, 179_999, 188_999, 191_999, 203_999, 206_999,
            209_999, 212_999, 215_999],
    "wikitext_103": [125_999, 128_999, 131_999, 134_999, 137_999, 152_999, 203_999, 212_999],
}


def main():
    snap, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    P, U, macro = load(snap)
    numbers = {}
    for s in ("ptb", "wikitext_103", "m2d2_wikipedia_unsplit"):
        steps = STEPS.get(s, STEPS["wikitext_103"])
        numbers[s] = {int(t): round(float(P.loc[t, s]), 4) for t in steps}
    flat = P.loc[155_999:188_999, "ptb"].drop(index=176_999)
    numbers["ptb_155k_188k_without_176999"] = {"min": round(float(flat.min()), 4), "max": round(float(flat.max()), 4)}
    print(json.dumps(numbers, indent=1))
    with open(os.path.join(out, "numbers_9874.json"), "w") as f:
        json.dump(numbers, f, indent=1)

    lo = 80_000
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    ax = axes[0]
    y = P.loc[lo:, "ptb"]
    ax.plot(y.index / 1e3, y.values, "-o", ms=3, color="black", lw=1.2, label="Paloma ptb")
    for (a, b), c in zip(READS.values(), ("#999999", "#e08a00", "#c0392b")):
        ax.axvspan(a / 1e3 - 0.25, b / 1e3 + 0.25, color=c, alpha=0.55, lw=0)
    for a, b in SCAN["ptb"]:
        ax.axvspan(a / 1e3, b / 1e3, color="#2e86de", alpha=0.18, lw=0)
    ax.text(103.5, 1.73, "scan\n102.0k–105.0k", color="#1f5fa8", ha="center", fontsize=9)
    ax.text(190.5, 1.73, "scan\n189.0k–192.0k", color="#1f5fa8", ha="center", fontsize=9)
    ax.text(176.2, 2.20, "torch-nplm\ntest split", color="#666666", ha="center", fontsize=8)
    ax.text(204.8, 2.20, "c28q1", color="#b06000", ha="center", fontsize=8)
    ax.text(213.0, 2.20, "torch-nplm\nPTB valid", color="#c0392b", ha="center", fontsize=8)
    ax.set_ylabel("ptb loss")
    ax.set_ylim(1.68, 2.27)
    ax.grid(alpha=0.3)
    ax.set_title("Paloma PTB and WikiText-103 on the hero. Shaded: reads mapped in #9874; blue: windows to scan")

    ax = axes[1]
    w = P.loc[lo:, "wikitext_103"]
    m = P.loc[lo:, "m2d2_wikipedia_unsplit"]
    ref = 125_999
    ax.plot(w.index / 1e3, w.values - w.loc[ref], "-o", ms=3, color="black", lw=1.2, label="wikitext_103")
    ax.plot(m.index / 1e3, m.values - m.loc[ref], "-o", ms=3, color="#16a085", lw=1.2,
            label="m2d2_wikipedia_unsplit (Wikipedia control)")
    for a, b in SCAN["wikitext_103"]:
        ax.axvspan(a / 1e3, b / 1e3, color="#2e86de", alpha=0.18, lw=0)
    ax.text(127.5, -0.235, "scan 126.0k–129.0k", color="#1f5fa8", ha="center", fontsize=9)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel(f"loss minus its value at step {ref:,}")
    ax.set_xlabel("step (thousands)")
    ax.set_ylim(-0.25, 0.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_9874.png"), dpi=150)
    print("wrote", os.path.join(out, "fig_9874.png"))


if __name__ == "__main__":
    main()
