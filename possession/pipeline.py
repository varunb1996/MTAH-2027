"""Data in, labels out. The one function run.py and tools/tune.py call."""

import numpy as np

from games import GAME
from . import atd, features, model, tagging
from .config import DEFAULT

MIN_PSEUDO = 500


class Prepared(object):
    """Everything that does not depend on the decoding parameters, computed once.

    Tuning re-runs only evidence() and decode() on top of this, which takes seconds.
    """

    def __init__(self, tracking, feats, tags, shape_model, shape_logit, n_pseudo):
        self.tracking = tracking
        self.features = feats
        self.tags = tags
        self.shape_model = shape_model
        self.shape_logit = shape_logit
        self.n_pseudo = n_pseudo


def prepare(params=DEFAULT, game=GAME, data_dir=None, use_cache=True, log=print):
    tracking = atd.load(game, data_dir, use_cache=use_cache)
    log("tracking: {} frames, {} slots, H = {}, A = {}".format(
        tracking.n_frames, len(tracking.team), tracking.home_name, tracking.away_name))

    feats = features.compute(tracking, params, game)
    log("ball observed on {:.1f}% of frames, known after gap-filling on {:.1f}%".format(
        100 * feats["observed"].mean(), 100 * feats["known"].mean()))

    tags = tagging.frame_signals(tracking.n_frames, tracking.home_name, tracking.away_name,
                                 params, game, data_dir)
    log("smart tagging: team prior on {:.1f}% of frames, set-piece window on {:.1f}%".format(
        100 * (tags[0] != 0).mean(), 100 * (tags[1] > 0).mean()))

    pseudo = model.pseudo_labels(feats)
    n_pseudo = int((pseudo != 0).sum())
    if n_pseudo < MIN_PSEUDO:
        # Too few clean control frames to fit anything trustworthy: run without the shape term.
        log("shape model: only {} self-labelled frames, disabled".format(n_pseudo))
        return Prepared(tracking, feats, tags, None, np.zeros(tracking.n_frames), n_pseudo)
    shape_model = model.ShapeModel().fit(feats["shape"], pseudo)
    shape_logit = shape_model.logit(feats["shape"])
    agree = np.mean(np.sign(shape_logit[pseudo != 0]) == pseudo[pseudo != 0])
    log("shape model: fitted on {} self-labelled frames, agrees with them on {:.1f}%".format(
        n_pseudo, 100 * agree))
    return Prepared(tracking, feats, tags, shape_model, shape_logit, n_pseudo)


def label(prepared, params=DEFAULT, game=GAME):
    """Full-length array of 'H' / 'A' / 'D' / 'X'."""
    team, dead = model.evidence(prepared.features, prepared.tags, params, prepared.shape_logit)
    return model.letters(model.decode(team, dead, params, game))


def summary(labels, game=GAME):
    """Possession count and mean duration over the graded halves -- no ground truth needed."""
    from games import half_video_blocks
    spells = []
    for _, first, last in half_video_blocks(game):
        seg = labels[first:last + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if seg[a] in ("H", "A"):
                spells.append(b - a)
    graded = np.concatenate([labels[f:l + 1] for _, f, l in half_video_blocks(game)])
    return {
        "possessions": len(spells),
        "mean_possession_s": float(np.mean(spells) / 25) if spells else 0.0,
        "share_H": float(np.mean(graded == "H")),
        "share_A": float(np.mean(graded == "A")),
        "share_D": float(np.mean(graded == "D")),
    }
