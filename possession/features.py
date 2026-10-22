"""Per-frame features from the tracking: what is near the ball, how the teams are shaped, whether
anyone is running. Nothing here makes a decision; model.py turns these into evidence.

All arrays are over video frames. NaN means "not observable on this frame".
"""

import numpy as np

from games import GAME, half_video_blocks

FPS = 25


def rolling_mean(x, window):
    """NaN-aware centred mean over `window` frames. Frames with no finite input stay NaN."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1:
        return x.copy()
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    half = window // 2
    index = np.arange(len(x))
    low = np.maximum(0, index - half)
    high = np.minimum(len(x), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / n, np.nan)


def percentile_rank(x, reference_mask):
    """Each frame's value as a fraction 0..1 of the reference frames' finite values below it."""
    ref = np.sort(x[reference_mask & np.isfinite(x)])
    if len(ref) == 0:
        return np.full(len(x), np.nan)
    rank = np.searchsorted(ref, np.where(np.isfinite(x), x, 0.0)) / float(len(ref))
    return np.where(np.isfinite(x), rank, np.nan)


def run_length(mask):
    """For each frame, how many consecutive True frames end there (0 where mask is False)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(len(mask), dtype=np.int64)
    count = 0
    for i, value in enumerate(mask):
        count = count + 1 if value else 0
        out[i] = count
    return out


def spread_forward(signal, window):
    """Carry each True frame of `signal` forward for `window` frames."""
    signal = np.asarray(signal, dtype=bool)
    cumulative = np.concatenate(([0], np.cumsum(signal)))
    index = np.arange(len(signal))
    low = np.maximum(0, index - window + 1)
    return (cumulative[index + 1] - cumulative[low]) > 0


def interpolate_gaps(x, max_gap):
    """Linear fill of NaN runs of at most max_gap frames that have a finite value on both sides."""
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(x))
    if len(finite) < 2:
        return x
    gaps = np.diff(finite)
    for k in np.flatnonzero((gaps > 1) & (gaps <= max_gap + 1)):
        a, b = finite[k], finite[k + 1]
        x[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
    return x


def _team_reduce(values, team_mask, how):
    """Per-frame min / mean / std over one team's tracked slots, NaN if none tracked."""
    sub = values[:, team_mask]
    finite = np.isfinite(sub)
    n = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "min":
            out = np.where(finite, sub, np.inf).min(axis=1)
            return np.where(n > 0, out, np.nan)
        if how == "max":
            out = np.where(finite, sub, -np.inf).max(axis=1)
            return np.where(n > 0, out, np.nan)
        total = np.where(finite, sub, 0.0).sum(axis=1)
        mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
        if how == "mean":
            return mean
        sq = np.where(finite, (sub - mean[:, None]) ** 2, 0.0).sum(axis=1)
        return np.where(n > 1, np.sqrt(sq / np.maximum(n, 1)), np.nan)


def attack_direction(tracking, game=GAME):
    """Per frame, +1 if the home team attacks towards x = length, -1 otherwise, 0 off the halves.

    Decided once per half from where each team's deepest player sits -- its goalkeeper -- which
    is far more stable than team centroids that move with play.
    """
    length = tracking.pitch[0]
    home = tracking.team == 0
    out = np.zeros(tracking.n_frames, dtype=np.int8)
    for _, first, last in half_video_blocks(game):
        sl = slice(first, last + 1)
        home_min = np.nanmedian(_team_reduce(tracking.px[sl], home, "min"))
        home_max = np.nanmedian(_team_reduce(tracking.px[sl], home, "max"))
        away_min = np.nanmedian(_team_reduce(tracking.px[sl], ~home, "min"))
        away_max = np.nanmedian(_team_reduce(tracking.px[sl], ~home, "max"))
        # Home keeper near x = 0 and away keeper near x = length -> home attacks +x.
        left = home_min + (length - away_max)
        right = (length - home_max) + away_min
        out[sl] = 1 if left <= right else -1
    return out


def compute(tracking, params, game=GAME):
    """Dict of per-frame feature arrays."""
    length, width = tracking.pitch
    n = tracking.n_frames
    home = tracking.team == 0
    px, py = tracking.px, tracking.py

    # -- ball ----------------------------------------------------------------------------------
    observed = np.isfinite(tracking.bx) & np.isfinite(tracking.by)
    bx = interpolate_gaps(tracking.bx, params.ball_gap_frames)
    by = interpolate_gaps(tracking.by, params.ball_gap_frames)
    known = np.isfinite(bx) & np.isfinite(by)
    ball_quality = np.where(observed, 1.0, np.where(known, params.interp_weight, 0.0))

    # Displacement across +/- 3 frames rather than frame to frame: detection jitter of a few tens
    # of centimetres would otherwise read as a ball travelling at 10 m/s while it sits at a
    # player's feet.
    reach = 3
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    dx[reach:-reach] = bx[2 * reach:] - bx[:-2 * reach]
    dy[reach:-reach] = by[2 * reach:] - by[:-2 * reach]
    ball_speed = np.clip(np.hypot(dx, dy) * FPS / (2 * reach), 0, 45)

    out_x = np.maximum(-tracking.bx, tracking.bx - length)
    out_y = np.maximum(-tracking.by, tracking.by - width)
    ball_out = observed & (np.fmax(out_x, out_y) > params.out_margin_m)

    # -- who is near it ------------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        dist = np.hypot(px - bx[:, None], py - by[:, None]).astype(np.float32)
    d_home = _team_reduce(dist, home, "min")
    d_away = _team_reduce(dist, ~home, "min")

    # -- movement ------------------------------------------------------------------------------
    speed = tracking.ps.astype(np.float64)
    if not np.isfinite(speed).any():
        # The export carries no speed channel: derive it from positions.
        speed = np.hypot(np.diff(px, axis=0, prepend=np.nan),
                         np.diff(py, axis=0, prepend=np.nan)) * FPS
    speed = np.where(speed < 12.0, speed, np.nan)  # identity switches look like teleports
    on_pitch = np.isfinite(px) & (px > -2) & (px < length + 2) & (py > -2) & (py < width + 2)
    n_players = on_pitch.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_speed = np.nansum(np.where(on_pitch, speed, np.nan), axis=1) / np.maximum(
            np.isfinite(np.where(on_pitch, speed, np.nan)).sum(axis=1), 1)
    mean_speed = np.where(n_players > 0, mean_speed, np.nan)

    # -- team shape ----------------------------------------------------------------------------
    direction = attack_direction(tracking, game).astype(np.float64)
    cx_h = _team_reduce(px, home, "mean")
    cx_a = _team_reduce(px, ~home, "mean")
    sx_h = _team_reduce(px, home, "std")
    sx_a = _team_reduce(px, ~home, "std")
    sy_h = _team_reduce(py, home, "std")
    sy_a = _team_reduce(py, ~home, "std")
    block = ((cx_h + cx_a) / 2 - length / 2) * direction
    block_velocity = np.diff(block, prepend=np.nan) * FPS

    smooth = max(1, int(params.shape_smooth_s * FPS))
    shape = np.column_stack([
        rolling_mean(sx_h - sx_a, smooth),
        rolling_mean(sy_h - sy_a, smooth),
        rolling_mean((cx_h - cx_a) * direction, smooth),
        rolling_mean(block, smooth),
        rolling_mean(np.clip(block_velocity, -10, 10), 2 * smooth),
        np.where(known, (bx - length / 2) * direction, 0.0),
    ])
    shape_names = ["len_diff", "width_diff", "centroid_gap", "block", "block_velocity",
                   "ball_forward"]

    graded = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        graded[first:last + 1] = True

    return {
        "graded": graded,
        "observed": observed,
        "known": known,
        "ball_quality": ball_quality,
        "ball_speed": ball_speed,
        "ball_out": ball_out,
        "d_home": d_home,
        "d_away": d_away,
        "mean_speed": mean_speed,
        "n_players": n_players,
        "direction": direction,
        "shape": shape,
        "shape_names": shape_names,
    }
