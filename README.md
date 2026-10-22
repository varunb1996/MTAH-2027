# Ball possession from tracking geometry — More than a Hack 2027, Phase 1

Who has the ball, on every frame of FC Barcelona vs Athletic Club: `H`, `A` or `D` (dead ball).

The approach in one line: per-frame evidence from ball control, team shape, player movement and the
Smart Tagging, decoded with a three-state Viterbi model whose transition costs make a possession an
interval rather than a string of independent frames. Details, assumptions and limitations are in
**[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**.

The organisers' original README, describing the challenge and the starter kit, is
[CHALLENGE_README.md](CHALLENGE_README.md).

## Reproduce the submission

Python 3.14.5 (anything from 3.9 should work), numpy and pandas only.

```bash
pip install -r requirements-solution.txt
```

Put the match bundle in `data/` (or point `MTAH_DATA_DIR` at it), laid out as
[DATA.md](DATA.md) describes. Only these three files are read; the videos and the broadcast
recording are not needed:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Then:

```bash
python run.py --team <team_name>
```

This writes `submission/<team_name>.csv`, runs the official validation on it, prints the
full-match possession statistics and scores it against the public five-minute ground truth. It
takes about 30 seconds the first time (parsing the raw tracking) and about 10 seconds afterwards,
when a parsed cache in `data/` is reused. The result is deterministic: `--no-cache` produces the
same file byte for byte.

Check the exact file you upload:

```bash
python evaluation/validate_submission.py submission/<team_name>.csv
```

## Layout

| path | what |
|---|---|
| `run.py` | data in, submission file out |
| `possession/config.py` | every parameter, with its reasoning |
| `possession/atd.py` | fast raw-ATD reader (pandas C parser) with a local cache |
| `possession/features.py` | per-frame ball, proximity, movement and team-shape features |
| `possession/tagging.py` | Smart Tagging as a weak team prior and a restart marker |
| `possession/model.py` | evidence, self-trained shape model, Viterbi decoding |
| `possession/pipeline.py` | glue, plus label-free possession statistics |
| `tools/tune.py` | one-parameter-at-a-time sensitivity sweep |
| `tools/make_synthetic_data.py` | a fake match in the bundle's formats, for testing without data |
| `evaluation/`, `starter/`, `games.py` | the organisers' scorer and starter kit, unchanged |

## Test without the data

```bash
python tools/make_synthetic_data.py /tmp/mtah_synthetic
MTAH_DATA_DIR=/tmp/mtah_synthetic python run.py --team synthetic --out /tmp/mtah_synthetic/out \
    --gt /tmp/mtah_synthetic/synthetic_gt.csv
```

## Data

The match data is licensed for this competition only and is not in this repository. `data/`,
`submission/`, the parsed cache and any zip are gitignored.
