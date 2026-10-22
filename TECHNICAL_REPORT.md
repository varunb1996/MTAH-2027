# Technical report — ball possession from tracking geometry

*More than a Hack 2027, Phase 1. Team: `<team_name>`.*

## 1. Approach

The task is to label each of 199,319 frames `H`, `A` or `D`. It is ranked on spellF1, which
scores whole possessions matched by temporal IoU. Two observations shaped everything:

1. **A possession is an interval, not a frame.** Per-frame classifiers flicker, and every flicker
   is a spurious possession. The labels therefore come from a sequence model whose output is a
   small number of long segments, not from smoothing a per-frame classifier afterwards.
2. **The ball is missing on 41% of graded frames**, disproportionately at the moments that
   matter. The model must carry possession through those gaps and still find hand-overs. So it
   needs evidence that does not depend on the ball (team shape, player movement) and a default
   that does the right thing when evidence is absent.

The model is a three-state hidden Markov model (H, A, D) decoded exactly with Viterbi, one half at
a time. Each frame contributes two log-odds:

- **team**: positive for home, negative for away;
- **dead**: positive for a dead ball;

combined as `log P(H) = log σ(−dead) + log 2σ(team)` and so on. Changing state has a fixed cost:
`switch_cost` between H and A, `to_dead_cost` / `from_dead_cost` into and out of D.

The transition cost is how the **"controlled possession"** rule of CHALLENGE.md is put into
practice. A team only takes the ball when its evidence, summed over consecutive frames, outweighs
the cost of a switch. With the defaults that means about 0.5 s of clean close control
(`switch_cost / ctrl_weight` = 12 frames). Three cases follow:

- A deflection or a touch without control does not pay for a switch.
- A player who keeps the ball does.
- A loose ball with no evidence leaves possession with the last team that had it, as the rules ask.

Decoding is offline (it sees the whole half), so transitions land where the evidence changes.
Nothing lags the way a sliding-window vote does, and the metric charges for lag heavily: being
1 s late costs over a third of spellF1.

We chose this over a learned frame classifier because of the labels available. There are 5 public
minutes and 17 possessions, which is too little to train or select anything with confidence
(EVALUATION.md shows this even for one smoothing parameter). The HMM has a few interpretable
parameters, each set from football reasoning rather than fitted to the public window.

## 2. Evidence

### Team evidence

**Ball control.** For every frame where the ball position is known, we take the distance from the
ball to the nearest home player and to the nearest away player. Evidence for a team is the product
of four factors:

- **Closeness:** full weight inside 1.5 m, falling to zero at 3.5 m.
- **Separation:** the nearest opponent must be at least 1.5 m further away for full weight.
- **Grounded:** the evidence fades out as ball speed rises from 8 to 16 m/s. A ball in flight past
  a player is a pass or a shot going by, not control.
- **Quality:** 1 for an observed ball, 0.5 for a ball interpolated across a gap of at most 12
  frames.

Ball speed is measured as displacement across ±3 frames. Frame-to-frame differences turn
detection jitter into 10 m/s of phantom speed.

**Team shape, self-trained.** When a team has the ball it spreads out and pushes forward, while the
defending team compacts behind the ball. We use six shape features:

- difference in length spread;
- difference in width spread;
- signed gap between the two team centroids;
- how far the whole block sits in either half;
- the block's velocity along the pitch;
- ball position along the pitch, when known.

Each is signed by the attacking direction, which is found per half from where each team's deepest
player (the goalkeeper) stands.

The shape model is a ridge logistic regression **fitted on this match without any ground truth**.
Its training labels are frames where the tracking shows unambiguous control: an observed ball
within 1 m of one team, the other team more than 2.5 m further away, a slow ball, and at least 8
consecutive such frames. The fitted model is then applied everywhere, which matters most on the
frames where the ball is lost. Its logit is clipped and down-weighted (`shape_weight` 0.15),
because neighbouring frames are not independent evidence.

**Smart Tagging.** Each phase tag carries a team, and we map it to a side:

- Attacking codes (Build Up, Progression, Creating Chances, …) vote for the tagged team.
- Defending codes (Defending in … third, Defensive Transition) vote for the other team.
- `N/A` and the throw-ins carry no vote.

The weight is deliberately small (0.05) because the tags are automatic and describe phases, not
possessions.

### Dead-ball evidence

- **Player movement.** We take the mean speed of tracked players, smoothed over 2 s and ranked
  against the whole match. Frames below the 40th percentile lean dead, those above lean alive.
  Ranking rather than a fixed m/s threshold keeps the rule independent of the tracker's speed
  calibration.
- **Ball off the pitch** by more than 0.5 m. This is strong evidence, and it is held for 3 s
  because the tracker usually loses a ball that has left play.
- **Ball parked**: observed and stationary for over 1 s, placed for a restart. In practice this
  almost never fires, because players walk with the ball before a restart rather than leaving it
  still. The next signal covers that case.
- **Ball carried slowly while everyone walks**: a player bringing the ball to a restart. Two
  conditions must hold for at least 2 s:
  - the ball's speed, averaged over 2 s, ranks in the slowest 30% of known-ball frames;
  - player movement ranks below the match median.

  On the public window it fires on 17% of dead-ball frames and on none of the live ones, so it
  carries a high weight (2.0).
- **Ball lost for over 2 s.** This is weak evidence on its own.
- **Smart Tagging set pieces**: from 4 s before the tag's start to 1 s after it.
- **Fewer than 12 tracked players**: the camera is not on play.

Two facts published by the organisers are used as hard constraints: each half's kickoff frame
(11,185 and 113,742). Graded frames before the second-half kickoff are dead ball.

## 3. Assets used and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD | yes | everything above |
| Smart Tagging | yes | weak team prior, restart marker |
| tactical homography | no | only needed to project onto the video, which we do not read |
| tactical video | no | the tracking already carries the geometry we use |
| broadcast video, ATD, homography | no | not frame-aligned to the tactical timeline; no mapping provided |

## 4. Possession rules implemented

- **Change of possession**: sustained close control by the other team, roughly 0.5 s or more of
  clean contact. The threshold is the ratio of the switch cost to the control weight.
- **Loose balls and aerial duels**: no evidence, so possession stays with the last team.
- **Deflections, touches, balls flying past**: too brief, or discounted by ball speed.
- **Restarts**: the decoder enters D from movement, ball-out and set-piece evidence. It leaves D
  for whichever team's evidence is stronger after the restart; nobody owns a dead ball.

## 5. Results

| | public spellF1 | public macroF1 | possessions in public window (truth: 17) |
|---|---|---|---|
| baseline_raw | 0.0378 | 0.5972 | 131 |
| baseline_smooth50 (published floor) | 0.3256 | 0.6207 | 26 |
| baseline_smooth125 | 0.4242 | 0.5812 | 16 |
| **ours** | **0.5241** | **0.8127** | 12 |

Over the full match, which is the only view that needs no ground truth, the submission has 276
possessions with a mean length of 14.0 s (the public truth's mean is 14.2 s). It splits the graded
time into 36.4% H, 24.4% A and 39.2% D.

Where it stands on the public window:

- **Teams are well separated.** About 90% of home frames and 93% of away frames get the right
  team.
- **Possessions are too few** (12 against 17). Short possessions of 1–3 s are merged into their
  neighbours. That is the price of a switch cost high enough to ignore deflections.

The public window is a correctness check, not a selection criterion. It has 17 possessions, and
one missed possession moves spellF1 by about 0.06. A 300-draw segment bootstrap of our spellF1
spans roughly 0.30–0.74. `tools/tune.py` shows the sensitivity to every parameter.

**Two changes were made after the first run on real data.**

The first added the slow-carry signal above. It caught 85 more dead-ball frames on the public
window and added no false ones. macroF1 went from 0.794 to 0.813; spellF1 did not change.

The second changed `dead_rank` from 0.30 to 0.40. The first run found only 135 of 1,469 dead-ball frames, and its full-match dead-ball share
was 29%. That is far below the roughly 40% that follows from typical effective playing time in
top-flight football (52–55 minutes of ball in play out of ~95). At 0.40 the full-match share is
37%. The new value was chosen on that football prior, not on the public score. Other settings
that score higher on the public window, such as a stronger dead prior or a lower switch cost,
were left alone because the bootstrap intervals overlap.

## 6. Limitations and what we would do next

- **Ball loss in crowded areas.** When the ball disappears in a duel and reappears with the other
  team, the decoder places the hand-over where the evidence changes. That can be late by the length
  of the gap. Using the homography and video to re-detect the ball in those gaps would help most.
- **Tracking identity switches** create position jumps. Speeds above 12 m/s are discarded, but a
  switch can still briefly put a "player" next to the ball.
- **Stoppages where the ball is lost or players keep moving.** About half of the public window's
  dead-ball frames are still missed, mostly in two cases:
  - the ball disappears for the whole stoppage;
  - players keep jogging, for example while repositioning for a corner.

  Recognising set-piece formations from the player geometry (a crowd in the box, one player at
  the corner flag) is the natural next signal.
- **Throw-ins carry no team** in the Smart Tagging. Recognising the thrower's team from the
  geometry at the touchline would sharpen restarts.
- **Fixed transition costs.** Costs that depend on context (lower near the ball, higher in
  midfield) would let short possessions through where they actually happen.
- **The broadcast feed is unused.** Aligning it to the tactical timeline would give a second view
  of the ball during occlusions.

## 7. Reproduction

```bash
pip install -r requirements-solution.txt
python run.py --team <team_name>
python evaluation/validate_submission.py submission/<team_name>.csv
```

The output is deterministic: there is no randomness anywhere, and `--no-cache` gives an identical
file.
