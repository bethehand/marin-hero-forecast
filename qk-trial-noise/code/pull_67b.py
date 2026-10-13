"""Pull the 67B qk test arms (20 steps from step 39k at 65K packing): train loss per step and the evals (read-only W&B)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
P = "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k_yarn_"
ARMS = ["mscale00", "mscale01", "mscale02", "long_mscale00", "long_mscale01", "long_mscale02"]
SUBSETS = ["4chan", "c4_100_domains", "c4_en", "dolma-v1_5", "dolma_100_programing_languages", "dolma_100_subreddits",
           "falcon-refinedweb", "gab", "m2d2_s2orc_unsplit", "m2d2_wikipedia_unsplit", "manosphere_meta_sep", "mc4", "ptb",
           "redpajama", "twitterAAE_HELM_fixed", "wikitext_103"]


def main():
    keys = [f"eval/paloma/{s}/loss" for s in SUBSETS] + ["eval/paloma/macro_loss", "eval/uncheatable_eval/macro_loss", "train/loss"]
    res = {}
    for arm in ARMS:
        name, rows = P + arm + "_test20", {}
        # sampledHistory returns only rows that contain every requested key, so query each key separately
        for k in keys:
            for r in wb.history(name, [k]):
                if r.get(k) is not None:
                    rows.setdefault(str(r["_step"]), {})[k] = r[k]
        res[name] = rows
        print(name, "steps:", len(rows))
    with open(os.path.join(OUT, "qk_arms_67b.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
