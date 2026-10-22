"""From features to labels: per-frame evidence, then one Viterbi pass per half.

Three hidden states, H, A and D. Each frame contributes two log-odds:

    team  -- positive for home, negative for away, from ball control, team shape and tagging
    dead  -- positive for a dead ball, from player movement, the ball off the pitch or parked,
             a long ball loss, tagged restarts

and P(H) = P(alive) P(home | alive), and so on. Changing state costs a fixed amount, so a team only
takes the ball when its evidence over several frames outweighs the cost. That one mechanism is the
"controlled possession" rule of CHALLENGE.md: a touch or a deflection does not pay for a switch,
a player keeping the ball does, and a loose ball with no evidence leaves possession where it was.
Decoding is offline, so transitions land where the evidence changes, with no smoothing lag.
"""

import numpy as np

from games import GAME, half_video_blocks
from . import features as feat

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D", "X"])
X = 3

# Published in FAQ.md / DATA.md: first frame of play in each half. The second half is graded from
# 110808 but kicks off at 113742, and every frame in between is dead ball.
KICKOFF_FRAMES = (11185, 113742)


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


# -- team shape model ------------------------------------------------------------------------------

def control_evidence(f, p):
    """Signed per-frame contact evidence: + home has the ball at its feet, - away does."""
    d_home, d_away = f["d_home"], f["d_away"]
    nearest = np.fmin(d_home, d_away)
    margin = np.abs(d_home - d_away)
    closeness = np.clip((p.ctrl_far_m - nearest) / (p.ctrl_far_m - p.ctrl_near_m), 0, 1)
    separation = np.clip(margin / p.ctrl_margin_m, 0, 1)
    grounded = np.clip((p.flight_speed_hi - f["ball_speed"])
                       / (p.flight_speed_hi - p.flight_speed_lo), 0, 1)
    grounded = np.where(np.isfinite(grounded), grounded, 1.0)
    sign = np.sign(d_away - d_home)
    out = sign * closeness * separation * grounded * f["ball_quality"]
    return np.where(np.isfinite(out), out, 0.0)


def pseudo_labels(f, min_run=8):
    """Frames where one team clearly has the ball at its feet: +1 home, -1 away, 0 unsure.

    Taken from the tracking alone -- no ground truth -- so the shape model can be fitted on the
    whole match rather than on the 7500 public frames.
    """
    d_home, d_away = f["d_home"], f["d_away"]
    clear = f["observed"] & (np.fmin(d_home, d_away) < 1.0) & (np.abs(d_home - d_away) > 2.5)
    clear &= np.where(np.isfinite(f["ball_speed"]), f["ball_speed"] < 6.0, False)
    label = np.where(clear, np.where(d_home < d_away, 1, -1), 0)
    # Keep only sustained stretches: a single close frame is often a pass going by.
    for value in (1, -1):
        mask = label == value
        lengths = feat.run_length(mask)
        ends = np.flatnonzero(mask & ~np.append(mask[1:], False))
        for end in ends:
            if lengths[end] < min_run:
                label[end - lengths[end] + 1:end + 1] = 0
    return label


class ShapeModel(object):
    """Ridge logistic regression on standardised shape features, fitted with Newton's method."""

    def __init__(self, l2=1.0):
        self.l2 = l2
        self.mean = self.scale = self.w = None

    def _design(self, shape):
        z = (shape - self.mean) / self.scale
        z = np.where(np.isfinite(z), z, 0.0)
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, shape, label):
        rows = label != 0
        self.mean = np.nanmean(shape[rows], axis=0)
        self.scale = np.nanstd(shape[rows], axis=0) + 1e-6
        Xm = self._design(shape[rows])
        y = (label[rows] > 0).astype(np.float64)
        # Balance the two teams so a possession-heavy side does not become the default.
        weight = np.where(y > 0, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
        w = np.zeros(Xm.shape[1])
        penalty = self.l2 * np.eye(len(w))
        penalty[0, 0] = 0.0
        for _ in range(25):
            prob = 1.0 / (1.0 + np.exp(-Xm @ w))
            grad = Xm.T @ (weight * (prob - y)) + penalty @ w
            hess = (Xm * (weight * prob * (1 - prob))[:, None]).T @ Xm + penalty
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def logit(self, shape):
        return self._design(shape) @ self.w


# -- evidence --------------------------------------------------------------------------------------

def evidence(f, tags, p, shape_logit):
    """(team_logodds, dead_logodds) per frame."""
    team_prior, set_piece = tags

    team = (p.ctrl_weight * control_evidence(f, p)
            + p.shape_weight * np.clip(shape_logit, -p.shape_clip, p.shape_clip)
            + p.tag_weight * team_prior)

    # Movement is judged against this match's own distribution rather than a fixed m/s figure,
    # so a tracker that reads systematically fast or slow does not move the threshold.
    speed = feat.rolling_mean(f["mean_speed"], int(p.dead_speed_smooth_s * feat.FPS))
    rank = feat.percentile_rank(speed, f["graded"])
    slow = np.clip((p.dead_rank - rank) / p.dead_rank_scale, -2, 2)
    slow = np.where(np.isfinite(slow), slow, 0.0)

    out = feat.spread_forward(f["ball_out"], int(p.out_hold_s * feat.FPS))
    missing = feat.run_length(~f["known"]) / (p.miss_after_s * feat.FPS)
    still = f["observed"] & np.where(np.isfinite(f["ball_speed"]),
                                     f["ball_speed"] < p.still_speed, False)
    parked = feat.run_length(still) >= int(p.still_after_s * feat.FPS)

    ball_speed = feat.rolling_mean(np.where(f["known"], f["ball_speed"], np.nan),
                                   int(p.carry_smooth_s * feat.FPS))
    ball_rank = feat.percentile_rank(ball_speed, f["graded"])
    slow_carry = (f["known"] & (ball_rank < p.carry_ball_rank)
                  & np.where(np.isfinite(rank), rank < p.carry_player_rank, False))
    carried = feat.run_length(slow_carry) >= int(p.carry_after_s * feat.FPS)

    few = f["n_players"] < p.few_players

    dead = (p.dead_prior
            + p.dead_speed_weight * slow
            + p.out_weight * out
            + p.miss_weight * np.clip(missing - 1.0, 0, 2)
            + p.still_weight * parked
            + p.carry_weight * carried
            + p.set_piece_weight * set_piece
            + p.few_players_weight * few)
    return team, dead


# -- decoding --------------------------------------------------------------------------------------

def viterbi(team, dead, p, start_dead=False):
    """Most likely H/A/D path for one half under fixed transition costs."""
    scale = p.evidence_scale
    alive = log_sigmoid(-dead)
    # The team term is a likelihood ratio, log(2 sigmoid), so that "no idea which team" costs an
    # alive state nothing. With plain log(sigmoid) every uninformative frame would charge both
    # alive states log 2 and quietly push the decoder towards D.
    emit = np.column_stack([
        alive + log_sigmoid(team) + np.log(2.0),
        alive + log_sigmoid(-team) + np.log(2.0),
        log_sigmoid(dead),
    ]) * scale
    cost = np.array([
        [0.0, p.switch_cost, p.to_dead_cost],
        [p.switch_cost, 0.0, p.to_dead_cost],
        [p.from_dead_cost, p.from_dead_cost, 0.0],
    ])
    n = len(team)
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    if start_dead:
        score[:2] -= 1e6
    neg_cost = -cost
    for t in range(1, n):
        cand = score[:, None] + neg_cost          # from (rows) -> to (columns)
        best = cand.argmax(axis=0)
        back[t] = best
        score = cand[best, [0, 1, 2]] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(team, dead, p, game=GAME):
    """Full-length label codes: H/A/D inside the halves, X outside them."""
    codes = np.full(len(team), X, dtype=np.int8)
    for (half, first, last), kickoff in zip(half_video_blocks(game), KICKOFF_FRAMES):
        start = max(first, kickoff)
        codes[first:start] = D
        codes[start:last + 1] = viterbi(team[start:last + 1], dead[start:last + 1], p)
    return codes


def letters(codes):
    return LETTERS[codes]
