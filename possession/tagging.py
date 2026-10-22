"""Smart Tagging as two per-frame signals: a weak team prior and a set-piece (restart) marker.

The tags are automatic and their intervals are phases, not possessions, so nothing here is copied
across as a label. They only nudge the evidence the sequence model decodes.
"""

import numpy as np

from games import GAME, require
from starter.load_smart_tagging import load as load_tags

# Which side a code's Team tag refers to: +1 the tagged team has the ball, -1 the tagged team is
# defending, so the other side has it. Codes not listed here carry no team prior.
CODE_SIDE = {
    "build up": 1,
    "progression": 1,
    "creating chances": 1,
    "ball in final third": 1,
    "ball in the box": 1,
    "players in the box": 1,
    "attacking transition": 1,
    "long ball": 1,
    "goals": 1,
    "defending in defensive third": -1,
    "defending in middle third": -1,
    "defending in attacking third": -1,
    "defensive transition": -1,
}
SET_PIECE_CODE = "set pieces"


def _norm(text):
    return (text or "").strip().lower()


def side_of_team_tag(value, home_name, away_name):
    """+1 home, -1 away, 0 unknown. The tag spells clubs differently from the ATD metadata
    ("Atletic Club"), so match on shared words rather than equality."""
    value = _norm(value)
    if not value or value == "n/a":
        return 0
    words = set(value.replace(".", " ").split()) - {"fc", "club", "cf", "de"}
    home = set(_norm(home_name).split()) - {"fc", "club", "cf", "de"}
    away = set(_norm(away_name).split()) - {"fc", "club", "cf", "de"}
    if words & home and not words & away:
        return 1
    if words & away and not words & home:
        return -1
    # Fallback published in DATA.md: Barcelona is home.
    return 1 if "barcelona" in value else -1


def frame_signals(n_frames, home_name, away_name, params, game=GAME, data_dir=None):
    """(team_prior, set_piece) arrays, or zeros if the tagging file is absent.

    team_prior  +1 / -1 / 0 per frame, the mean of the overlapping tags' votes
    set_piece   1.0 on frames leaning dead around a tagged restart
    """
    team = np.zeros(n_frames, dtype=np.float32)
    votes = np.zeros(n_frames, dtype=np.float32)
    set_piece = np.zeros(n_frames, dtype=np.float32)
    try:
        require(["smart_tagging"], game, data_dir)
    except IOError:
        return team, set_piece

    fps = 25
    for event in load_tags(game, data_dir):
        code = _norm(event["code"])
        start = max(0, event["start_frame"])
        end = min(n_frames, max(start + 1, event["end_frame"]))
        if code == SET_PIECE_CODE:
            lo = max(0, start - int(params.set_piece_pre_s * fps))
            hi = min(n_frames, start + int(params.set_piece_post_s * fps))
            set_piece[lo:hi] = 1.0
        side = CODE_SIDE.get(code, 0)
        team_side = side_of_team_tag(event["tags"].get("Team"), home_name, away_name)
        if side and team_side:
            team[start:end] += side * team_side
            votes[start:end] += 1
    team = np.where(votes > 0, team / np.maximum(votes, 1), 0).astype(np.float32)
    return team, set_piece
