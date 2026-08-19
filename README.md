# Precision Platformer Gauntlet

I have created deterministic pygame platformer built as a Gymnasium environment, designed
to make an RL agent's improvement over training *visually obvious*. The level
is a fixed sequence of 40 jumps where the gap size and platform width scale
on an exponential curve — brutal-looking by the end, but altered to always stay
just barely possible.

## How the difficulty curve works

- **Gap size** grows from `70px` and asymptotically approaches `90%` of the
  agent's theoretical max jump distance (`~216px`, derived from jump velocity,
  gravity, and horizontal speed). It never quite reaches the true max, just makes it extremely difficult.
- **Platform width** shrinks from `150px` down to a floor of `46px`
  (roughly 1.8x the agent's own width), making jumps for the agent harder.

Starting at platform 14, some platforms also drift vertically (sine wave),
adding a timing element on top of the spacing/width difficulty. In v4, I added a toggle for this, where you can set moving_platforms to true or false.

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

Requirements:
pygame>=2.5.0
gymnasium>=1.0.0
stable-baselines3>=2.4.0
torch>=2.2.0
numpy>=1.24,<3.0
matplotlib>=3.7
tensorboard>=2.15
tqdm>=4.66
rich>=13.7
Pillow>=10.0

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

## Play the platformer, or visualize the agents progress:

```bash
python launcher.py
```
you have the option to see the agent play at certain checkpoints, play yourself, and see the graph of the agent's improvement over iterations.

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
