"""Read the raw ATD into dense arrays, fast, and cache them.

starter/baseline.py streams the file line by line in pure Python. That is clear but slow, and this
pipeline needs every slot of every frame, not just the nearest one. Here the whole file goes
through pandas' C parser in one call, after turning both separators into commas.

Raw line:  "<frame_id>:<slot>;<slot>;...;<slot>:<ball>"
           each slot "x,y,speed", the ball "x,y[,...]", NaN for a missing observation.

The result is cached next to the data as an .npz, so later runs take a second. The cache lives in
data/, which is gitignored: it is derived frame-level tracking and must not be published.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, require, source_frames

HOME, AWAY = 0, 1
CACHE_VERSION = 1


class Tracking(object):
    """Dense per-frame arrays, indexed by video frame (row i = submission row i).

    px, py, ps  (n_frames, n_slots) float32, metres and m/s, NaN where the slot is not tracked
    bx, by      (n_frames,) float32, metres, NaN where the ball is not tracked
    team        (n_slots,) int8, HOME or AWAY
    """

    def __init__(self, px, py, ps, bx, by, team, home_name, away_name, pitch):
        self.px, self.py, self.ps = px, py, ps
        self.bx, self.by = bx, by
        self.team = team
        self.home_name, self.away_name = home_name, away_name
        self.pitch = pitch

    @property
    def n_frames(self):
        return len(self.bx)


def read_meta(meta_path):
    """(team per slot, home name, away name). First team in the file is home, as in baseline.py."""
    root = ET.parse(meta_path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0][0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    team = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return team, teams[0][1] or "home", teams[1][1] or "away"


def _layout(first_line):
    """Column index of every slot's x and the ball's x, once both separators are commas.

    Read off the first line so that an empty trailing group, or a ball with extra fields, shifts
    nothing. Every line is assumed to share the layout; parse() checks the column count.
    """
    parts = first_line.rstrip("\r\n").split(":")
    if len(parts) != 3:
        raise ValueError("unexpected ATD line layout: {} ':'-separated parts".format(len(parts)))
    col = 1
    slot_cols, slot_widths = [], []
    for group in parts[1].split(";"):
        width = len(group.split(","))
        if group:
            slot_cols.append(col)
            slot_widths.append(width)
        col += width
    ball_col = col
    n_cols = col + len(parts[2].split(","))
    return slot_cols, slot_widths, ball_col, n_cols


def parse(raw_path, n_slots, n_frames):
    """(frame_id, slots[n, n_slots, 3], ball[n, 2]) in normalised units, NaN where missing."""
    with open(raw_path) as handle:
        first = handle.readline()
    slot_cols, slot_widths, ball_col, n_cols = _layout(first)
    if len(slot_cols) != n_slots:
        raise ValueError("ATD has {} slot groups per line, metadata lists {}".format(
            len(slot_cols), n_slots))

    with open(raw_path, "rb") as handle:
        text = handle.read().replace(b":", b",").replace(b";", b",")
    table = pd.read_csv(io.BytesIO(text), header=None, names=range(n_cols), dtype=np.float32,
                        na_values=["NaN", "nan", ""], keep_default_na=False, engine="c")
    del text
    values = table.to_numpy(dtype=np.float32)
    del table

    frame_id = values[:, 0].astype(np.int64)
    slots = np.full((len(values), n_slots, 3), np.nan, dtype=np.float32)
    for k, (col, width) in enumerate(zip(slot_cols, slot_widths)):
        slots[:, k, :min(width, 3)] = values[:, col:col + min(width, 3)]
    ball = values[:, ball_col:ball_col + 2]
    return frame_id, slots, ball


def load(game=GAME, data_dir=None, use_cache=True):
    """The tactical ATD as a Tracking, from cache when possible."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team, home_name, away_name = read_meta(meta_path)

    cache = os.path.join(os.path.dirname(raw_path), "cache_possession_atd.npz")
    stamp = np.array([CACHE_VERSION, os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            if np.array_equal(z["stamp"], stamp):
                return Tracking(z["px"], z["py"], z["ps"], z["bx"], z["by"], team,
                                home_name, away_name, (length, width))

    frame_id, slots, ball = parse(raw_path, len(team), n_frames)
    rows = frame_id - 1  # ATD counts from 1, submissions from 0
    keep = (rows >= 0) & (rows < n_frames)

    px = np.full((n_frames, len(team)), np.nan, dtype=np.float32)
    py, ps = px.copy(), px.copy()
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = bx.copy()
    px[rows[keep]] = slots[keep, :, 0] * length
    py[rows[keep]] = slots[keep, :, 1] * width
    ps[rows[keep]] = slots[keep, :, 2]
    bx[rows[keep]] = ball[keep, 0] * length
    by[rows[keep]] = ball[keep, 1] * width

    if use_cache:
        np.savez(cache, stamp=stamp, px=px, py=py, ps=ps, bx=bx, by=by)
    return Tracking(px, py, ps, bx, by, team, home_name, away_name, (length, width))
