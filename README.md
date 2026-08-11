# Precision Platformer Gauntlet

A deterministic pygame platformer built as a Gymnasium environment, designed
to make an RL agent's improvement over training *visually obvious*. The level
is a fixed sequence of ~40 jumps where the gap size and platform width scale
on an exponential curve — brutal-looking by the end, but tuned to always stay
just barely possible.

## How the difficulty curve works

Two things escalate exponentially as the level progresses:

- **Gap size** grows from `70px` and asymptotically approaches `90%` of the
  agent's theoretical max jump distance (`~216px`, derived from jump velocity,
  gravity, and horizontal speed). It never quite reaches the true max, so
  every jump stays physically completable — it just requires increasingly
  precise timing and full-speed horizontal movement.
- **Platform width** shrinks from `150px` down to a floor of `46px`
  (roughly 1.8x the agent's own width), so late-level landings require real
  precision, not just "jump in the general direction."

Starting at platform 14, some platforms also drift vertically (sine wave),
adding a timing element on top of the spacing/width difficulty.

Because gap and width are both deterministic functions of platform index
(seeded, not random noise), every training run faces the *exact same*
gauntlet — so "furthest platform reached" is a clean, comparable metric
across episodes and training runs.

## Files

| File | Purpose |
|---|---|
| `platformer_env.py` | The Gymnasium environment: physics, level generation, rendering |
| `train.py` | Trains a PPO agent (Stable-Baselines3), logs progress to CSV |
| `play.py` | Loads a trained model and plays it back in a live pygame window |
| `plot_progress.py` | Plots furthest-platform-reached and reward over training |

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --timesteps 2000000 --n-envs 8
```

This runs 8 parallel environments, checkpoints the model every ~50k steps
into `checkpoints/`, and logs one row per finished episode to
`training_log.csv` (episode number, timesteps so far, furthest platform
reached, episode reward). Training for ~1-2M timesteps is usually enough to
see the agent go from falling at platform 2-3 to clearing most or all of
the level.

To resume a previous run:

```bash
python train.py --timesteps 1000000 --resume
```

## Watch it play

```bash
python play.py --model platformer_model.zip
```

Or watch an earlier checkpoint to compare against the final model and show
the improvement directly:

```bash
python play.py --model checkpoints/ppo_platformer_100000_steps.zip
python play.py --model checkpoints/ppo_platformer_1500000_steps.zip
```

Playing the same seed at different checkpoints side by side (record both,
edit together) is the most convincing way to show "before vs. after" to
a viewer.

## Visualize the learning curve

```bash
python plot_progress.py
```

Produces `progress.png` with two panels: furthest platform reached per
episode (with a rolling average climbing toward the top of the level), and
episode reward over time.

## Tuning knobs (in `platformer_env.py`)

- `GAP_GROWTH` — higher = gaps ramp up to max difficulty faster
- `WIDTH_SHRINK` — higher = platforms shrink faster
- `MOVING_PLATFORM_START_INDEX` / `MOVING_PLATFORM_AMPLITUDE` — when moving
  platforms kick in and how far they drift
- `NUM_PLATFORMS` — level length
- `MAX_EPISODE_STEPS` — timeout per episode (currently ~30s at 60fps)

If early training looks like the agent can never progress at all, the two
most common fixes are lowering `GAP_GROWTH` slightly (easing the curve) or
increasing `ent_coef` in `train.py` (more exploration).

## A note on this sandbox

This project was scaffolded without network access, so the files are
syntax-checked (`py_compile`) but not run end-to-end here — you'll want to
do a first `python train.py --timesteps 20000 --n-envs 4` smoke test on your
own machine to confirm environment behavior before committing to a long run.
