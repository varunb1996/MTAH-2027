"""Sensitivity sweep: move one parameter at a time around the defaults and report what changes.

    python tools/tune.py                         # against the public five minutes
    python tools/tune.py --gt path/to/gt.csv     # against any frame,label file

For every value it prints the public-window spellF1 and macroF1 next to two numbers that need no
ground truth at all: how many possessions the full match comes out with, and their mean length.

Read it the way EVALUATION.md asks. Seventeen public possessions cannot choose between two close
settings -- one missed possession moves spellF1 by about 0.06. What the sweep is good for:
spotting a parameter the result is fragile to, and keeping the full-match possession statistics
football-plausible (real matches have possessions of roughly 10-15 s on average).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import ROOT, pipeline  # noqa: E402
from possession.config import DEFAULT  # noqa: E402
import possession_metrics as pm  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

SWEEP = {
    "switch_cost": [6, 9, 12, 18, 24],
    "to_dead_cost": [8, 12, 15, 22, 30],
    "from_dead_cost": [8, 12, 15, 22, 30],
    "ctrl_weight": [0.5, 0.75, 1.0, 1.5, 2.0],
    "shape_weight": [0.0, 0.05, 0.15, 0.3, 0.5],
    "tag_weight": [0.0, 0.05, 0.15, 0.3],
    "dead_rank": [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "dead_rank_scale": [0.05, 0.1, 0.15, 0.2],
    "dead_speed_weight": [0.5, 1.0, 1.5, 2.0],
    "dead_speed_smooth_s": [1.0, 2.0, 4.0, 6.0],
    "dead_prior": [-2.0, -1.5, -1.0, -0.5, 0.0],
    "out_weight": [0.0, 1.5, 3.0, 5.0],
    "set_piece_weight": [0.0, 0.25, 0.5, 1.0],
    "ctrl_near_m": [1.0, 1.5, 2.0, 2.5],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=PUBLIC_GT)
    parser.add_argument("--only", nargs="+", help="sweep just these parameters")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="segment-bootstrap draws for the spellF1 interval of the defaults")
    args = parser.parse_args()

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    prepared = pipeline.prepare(DEFAULT)

    def run(params, bootstrap=0):
        labels = pipeline.label(prepared, params)
        result = pm.evaluate(gt, pm.label_codes(np.asarray(labels, dtype=object)),
                             bootstrap_n=bootstrap)
        return result, pipeline.summary(labels)

    base, base_stats = run(DEFAULT, args.bootstrap)
    print("\ndefaults: spellF1 {:.4f}  macroF1 {:.4f}  | full match {} possessions, {:.1f} s"
          .format(base["spell_f1_mean"], base["macro_f1"], base_stats["possessions"],
                  base_stats["mean_possession_s"]))
    if args.bootstrap:
        print("  spellF1 bootstrap interval {:.4f} .. {:.4f}".format(
            base["spell_bootstrap_low"], base["spell_bootstrap_high"]))

    print("\n{:<18} {:>8}  {:>8} {:>8}  {:>6} {:>7}".format(
        "parameter", "value", "spellF1", "macroF1", "poss", "mean_s"))
    for name, values in SWEEP.items():
        if args.only and name not in args.only:
            continue
        for value in values:
            result, stats = run(DEFAULT.with_(**{name: value}))
            mark = "*" if value == getattr(DEFAULT, name) else " "
            print("{:<18} {:>7}{} {:>8.4f} {:>8.4f}  {:>6} {:>7.1f}".format(
                name, value, mark, result["spell_f1_mean"], result["macro_f1"],
                stats["possessions"], stats["mean_possession_s"]))
        print()


if __name__ == "__main__":
    main()
