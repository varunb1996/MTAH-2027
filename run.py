"""Regenerate the submission file from the data bundle.

    python run.py --team <team_name>

writes submission/<team_name>.csv, runs the official validation on it, and scores it against the
public five-minute ground truth. Deterministic: the same data gives the same file, byte for byte.
"""

import argparse
import os

import numpy as np

from possession import ROOT, pipeline
from possession.config import DEFAULT
import possession_metrics as pm
from games import GAME, GAMES

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def write(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path):
    reason = pm.validate_submission(pm.read_label_csv(path), GAMES[GAME]["n_video_frames"])
    return reason


def score(labels, gt_path=PUBLIC_GT):
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    return pm.evaluate(gt, pm.label_codes(np.asarray(labels, dtype=object)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default="submission", help="file name: <team>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "submission"))
    parser.add_argument("--no-cache", action="store_true", help="re-read the raw ATD")
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground truth to report against (default: the public 5 minutes)")
    args = parser.parse_args()

    prepared = pipeline.prepare(DEFAULT, use_cache=not args.no_cache)
    labels = pipeline.label(prepared, DEFAULT)

    path = write(os.path.join(args.out, "{}.csv".format(args.team)), labels)
    reason = validate(path)
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    print("validation: {}".format("OK" if reason is None else "FAIL -- " + reason))

    stats = pipeline.summary(labels)
    print("full match: {possessions} possessions, mean {mean_possession_s:.1f} s, "
          "H {share_H:.1%}  A {share_A:.1%}  D {share_D:.1%}".format(**stats))

    result = score(labels, args.gt)
    print("{}: spellF1".format(os.path.basename(args.gt)) + " {:.4f}  macroF1 {:.4f}  possessions {} vs {} in the truth".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
        result["n_gt_spells"]))
    print("  (a sanity check, not a leaderboard -- see EVALUATION.md)")


if __name__ == "__main__":
    main()
