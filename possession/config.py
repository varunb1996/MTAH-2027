"""Every tunable number in the pipeline, in one place.

Units are metres, seconds or frames (25 fps) as the name says. Weights are in log-odds: a weight
of 1.0 shifts a per-frame probability from 0.5 to 0.73.

The defaults are set from football reasoning, not fitted to the public five minutes -- see
EVALUATION.md for why tuning against 17 possessions picks the wrong answer. tools/tune.py reports
how sensitive the result is to each of them.
"""

from dataclasses import dataclass, asdict, replace

FPS = 25


@dataclass(frozen=True)
class Params:
    # -- ball track ----------------------------------------------------------------------------
    # Ball gaps up to this many frames are bridged by linear interpolation. Longer gaps stay
    # unknown: a ball missing for half a second may have been passed anywhere.
    ball_gap_frames: int = 12
    # Interpolated ball positions count for this fraction of an observed one.
    interp_weight: float = 0.5

    # -- control evidence ----------------------------------------------------------------------
    # A player inside ctrl_near_m of the ball is in contact; the evidence fades to nothing at
    # ctrl_far_m. The nearest opponent has to be further away by ctrl_margin_m for full weight.
    ctrl_near_m: float = 1.5
    ctrl_far_m: float = 3.5
    ctrl_margin_m: float = 1.5
    # Log-odds per frame of clear contact. Together with switch_cost this sets how long a team
    # has to hold the ball before the model believes possession changed: switch_cost / ctrl_weight
    # frames of clean contact. 12 / 1.0 = 0.5 s, which separates a controlled touch from a
    # deflection.
    ctrl_weight: float = 1.0
    # A ball moving faster than this is in flight (pass, shot, clearance): a player it passes is
    # not controlling it. Evidence fades between the two speeds.
    flight_speed_lo: float = 8.0
    flight_speed_hi: float = 16.0

    # -- team shape ----------------------------------------------------------------------------
    # The shape model is fitted per match on frames where the ball is clearly under one team's
    # control (no ground truth involved), then applied everywhere -- most usefully on the 41% of
    # frames where the ball is lost. Its logit is clipped and scaled because neighbouring frames
    # are not independent evidence.
    shape_weight: float = 0.15
    shape_clip: float = 3.0
    shape_smooth_s: float = 1.0

    # -- smart tagging -------------------------------------------------------------------------
    tag_weight: float = 0.05

    # -- dead ball -----------------------------------------------------------------------------
    # Prior log-odds that the ball is dead. Roughly 30% of graded frames are dead ball.
    dead_prior: float = -1.0
    # Players walk during stoppages. The mean speed of tracked players, smoothed over
    # dead_speed_smooth_s, is ranked against the whole match's graded frames: below the
    # dead_rank percentile the frame looks dead, above it alive. Top-flight matches have roughly
    # 52-55 minutes of ball in play out of ~95, so around 40% of graded time is dead ball. At 0.40
    # the full match comes out 37% dead; at the first guess, 0.30, it was 29%, too few.
    dead_rank: float = 0.40
    dead_rank_scale: float = 0.10
    dead_speed_weight: float = 1.0
    dead_speed_smooth_s: float = 2.0
    # The ball observed off the pitch by more than out_margin_m.
    out_margin_m: float = 0.5
    out_weight: float = 3.0
    # How long the ball can be out before the smoothing loses it: the out-of-pitch evidence is
    # spread forward over this window, because the tracker usually drops a ball that left play.
    out_hold_s: float = 3.0
    # Ball lost for a long stretch.
    miss_after_s: float = 2.0
    miss_weight: float = 0.5
    # Ball observed and stationary: placed for a restart.
    still_speed: float = 0.7
    still_after_s: float = 1.0
    still_weight: float = 1.0
    # Ball carried slowly while everyone walks: a player bringing the ball to a restart. The ball's
    # speed, averaged over carry_smooth_s, ranks below carry_ball_rank of the match's known-ball
    # frames while player movement ranks below carry_player_rank, for at least carry_after_s.
    # Both are ranks for the same reason as dead_rank: no dependence on the tracker's calibration.
    carry_ball_rank: float = 0.30
    carry_player_rank: float = 0.50
    carry_smooth_s: float = 2.0
    carry_after_s: float = 2.0
    # Strong, because the signal is precise: on the public window it fires on 17% of dead-ball
    # frames and on none of the live ones.
    carry_weight: float = 2.0
    # Smart Tagging set pieces: frames from set_piece_pre_s before the tag's start to
    # set_piece_post_s after it lean dead (the restart happens inside the tag).
    set_piece_pre_s: float = 4.0
    set_piece_post_s: float = 1.0
    set_piece_weight: float = 0.5
    # Fewer tracked players than this means the camera is not on play.
    few_players: int = 12
    few_players_weight: float = 2.0

    # -- sequence model ------------------------------------------------------------------------
    # Cost, in log-odds, of each kind of transition. Higher costs mean fewer, longer segments.
    switch_cost: float = 12.0
    to_dead_cost: float = 15.0
    from_dead_cost: float = 15.0
    # Each frame's evidence is multiplied by this before decoding. Lower values trust the
    # transition costs more relative to the evidence.
    evidence_scale: float = 1.0

    def to_dict(self):
        return asdict(self)

    def with_(self, **changes):
        return replace(self, **changes)


DEFAULT = Params()
