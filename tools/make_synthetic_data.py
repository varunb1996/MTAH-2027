"""Write a fake match in the bundle's file formats, for testing the pipeline without the real data.

    python tools/make_synthetic_data.py <out_dir>
    MTAH_DATA_DIR=<out_dir> python run.py

It produces data/tactical/{_atd.xml, _atd.txt, _smart_tagging.xml} with the same layout as the
real bundle, plus synthetic_gt.csv, the labels the fake match was generated from. The football is
crude -- players drift around formation spots, the ball hops between teammates -- but the things
the pipeline relies on are there: possession changes, stoppages with the ball off the pitch, the
ball missing for long stretches, and players slowing down when play stops.

This is a smoke test, not a benchmark. Scores on it say the plumbing works, nothing more.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import GAMES, half_video_blocks  # noqa: E402

N_FRAMES = GAMES["BAR-ATH"]["n_video_frames"]
LENGTH, WIDTH = GAMES["BAR-ATH"]["pitch"]
N_SLOTS = 46
FPS = 25
KICKOFFS = (11185, 113742)

FORMATION = np.array([[5, 34], [20, 10], [18, 27], [18, 41], [20, 58], [35, 15], [33, 34],
                      [35, 53], [48, 12], [50, 34], [48, 56]], dtype=float)


def simulate_states(rng, first, last, kickoff):
    """Per-frame 0 = H, 1 = A, 2 = D for one half."""
    states = np.full(last - first + 1, 2, dtype=np.int8)
    t, team = kickoff - first, rng.integers(2)
    while t < len(states):
        n = int(max(1.0, rng.exponential(12.0)) * FPS)
        states[t:t + n] = team
        t += n
        if rng.random() < 0.3:
            n = int(rng.uniform(5, 20) * FPS)
            states[t:t + n] = 2
            t += n
            team = rng.integers(2)
        else:
            team = 1 - team
    return states


def simulate(rng):
    truth = np.full(N_FRAMES, "X", dtype="<U1")
    px = np.full((N_FRAMES, N_SLOTS), np.nan)
    py = px.copy()
    ps = px.copy()
    bx = np.full(N_FRAMES, np.nan)
    by = bx.copy()

    for (half, first, last), kickoff in zip(half_video_blocks("BAR-ATH"), KICKOFFS):
        states = simulate_states(rng, first, last, kickoff)
        truth[first:last + 1] = np.array(["H", "A", "D"])[states]
        home_dir = 1 if half == 1 else -1
        offset = np.zeros((22, 2))
        holder, pass_left, dead_for = 0, 0, 0
        ball = np.array([LENGTH / 2, WIDTH / 2])
        prev = None
        for i, s in enumerate(states):
            f = first + i
            alive = s < 2
            # Formation spots: home defends x=0 in half 1, attacking side pushes forward.
            push = np.zeros(22)
            if alive:
                push[:11] = 12 if s == 0 else -6
                push[11:] = 12 if s == 1 else -6
            home = FORMATION.copy()
            home[:, 0] = home[:, 0] + push[:11]
            away = FORMATION.copy()
            away[:, 0] = away[:, 0] + push[11:]
            if home_dir < 0:
                home[:, 0] = LENGTH - home[:, 0]
            else:
                away[:, 0] = LENGTH - away[:, 0]
            spots = np.vstack([home, away])
            offset += rng.normal(0, 0.15 if alive else 0.05, offset.shape)
            offset *= 0.995
            pos = spots + offset * 8
            if prev is not None:
                pos = prev + np.clip(pos - prev, -0.25 if alive else 0.06,
                                     0.25 if alive else 0.06)
            speed = np.hypot(*(pos - prev).T) * FPS if prev is not None else np.zeros(22)
            prev = pos

            if alive:
                team_players = np.arange(11) + (11 if s == 1 else 0)
                if holder not in team_players or (pass_left == 0 and rng.random() < 1 / 50):
                    holder = int(rng.choice(team_players))
                    pass_left = 12
                if pass_left > 0:
                    a = 1 - pass_left / 12.0
                    ball = (1 - a) * ball + a * pos[holder]
                    pass_left -= 1
                else:
                    ball = pos[holder] + rng.normal(0, 0.3, 2)
            else:
                # Off the pitch for two seconds, then placed on the line for the restart.
                dead_for = dead_for + 1 if i and states[i - 1] == 2 else 0
                ball = np.array([LENGTH + 2.0 if dead_for < 50 else LENGTH - 0.3, ball[1]])

            slots = list(range(11)) + list(range(23, 34))
            px[f, slots], py[f, slots], ps[f, slots] = pos[:, 0], pos[:, 1], speed
            bx[f], by[f] = ball

        # The ball goes missing in chunks, more often while it is in play.
        i = first
        while i <= last:
            if rng.random() < 0.012:
                n = int(rng.exponential(30))
                bx[i:i + n] = by[i:i + n] = np.nan
                i += n
            i += 1
    return truth, px, py, ps, bx, by


def write_meta(path):
    teams = '<Team id="T1"><Name>FC Barcelona</Name></Team><Team id="T2"><Name>Athletic Club</Name></Team>'
    players = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}'
                      '</ShirtNumber></Player>'.format(k + 1, "T1" if k < 23 else "T2")
                      for k in range(N_SLOTS))
    with open(path, "w") as handle:
        handle.write('<?xml version="1.0"?><Main><Metadata><Teams>{}</Teams><Players>{}'
                     '</Players></Metadata></Main>'.format(teams, players))


def fmt(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def write_raw(path, px, py, ps, bx, by):
    with open(path, "w") as handle:
        for f in range(N_FRAMES):
            groups = []
            for k in range(N_SLOTS):
                if np.isfinite(px[f, k]):
                    groups.append("{:.5f},{:.5f},{:.2f}".format(px[f, k] / LENGTH,
                                                                py[f, k] / WIDTH, ps[f, k]))
                else:
                    groups.append("NaN,NaN,NaN")
            ball = "{},{},0".format(fmt(bx[f] / LENGTH), fmt(by[f] / WIDTH))
            handle.write("{}:{}:{}\n".format(f + 1, ";".join(groups), ball))


def write_tagging(path, truth):
    """Set-piece tags at each restart, and a team phase tag per possession, both a little off."""
    rng = np.random.default_rng(1)
    rows = []
    change = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(truth)]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if truth[a] in "HA":
            team = "Barcelona" if truth[a] == "H" else "Atletic Club"
            jitter = rng.normal(0, 1.0)
            rows.append(("BUILD UP", (a / FPS) + jitter, (b / FPS) + jitter, team))
            if a > 0 and truth[a - 1] == "D":
                rows.append(("SET PIECES", a / FPS - 2, a / FPS + 3, "N/A"))
    with open(path, "w") as handle:
        handle.write("<file><ALL_INSTANCES>")
        for k, (code, start, end, team) in enumerate(rows):
            handle.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}"
                         "</code><label><group>Team</group><text>{}</text></label></instance>"
                         .format(k, max(start, 0), max(end, 0), code, team))
        handle.write("</ALL_INSTANCES></file>")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    folder = os.path.join(args.out, "tactical")
    os.makedirs(folder, exist_ok=True)
    truth, px, py, ps, bx, by = simulate(np.random.default_rng(args.seed))
    write_meta(os.path.join(folder, "BAR-ATH_tactical_atd.xml"))
    write_raw(os.path.join(folder, "BAR-ATH_tactical_atd.txt"), px, py, ps, bx, by)
    write_tagging(os.path.join(folder, "BAR-ATH_tactical_smart_tagging.xml"), truth)
    with open(os.path.join(args.out, "synthetic_gt.csv"), "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, v) for i, v in enumerate(truth))
    print("wrote synthetic match to {}".format(args.out))


if __name__ == "__main__":
    main()
