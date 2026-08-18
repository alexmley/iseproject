"""
Train a PPO agent on the Precision Platformer Gauntlet.

Changes vs v1/v2:
  - Uses DummyVecEnv by default (SubprocVecEnv silently fails on macOS
    with newer Python due to 'spawn' multiprocessing mode)
  - ent_coef=0.08   (high entropy forces exploration, prevents collapse)
  - Linear LR decay 3e-4 → 1e-5
  - Compatible with SB3 2.4 through 2.9 and gymnasium 1.0 through 1.3

Usage:
    python train.py --timesteps 3000000 --n-envs 8
    python train.py --timesteps 1000000 --n-envs 4 --resume
"""

import argparse
import csv
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from platformer_env import PlatformerGauntletEnv

LOG_PATH       = "training_log.csv"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH     = "platformer_model.zip"


def make_env(seed):
    def _init():
        env = PlatformerGauntletEnv(render_mode=None, seed=seed)
        return Monitor(env)
    return _init


class ProgressLoggerCallback(BaseCallback):
    """Logs furthest platform reached per episode to CSV."""

    def __init__(self, log_path):
        super().__init__()
        self.log_path      = log_path
        self.episode_count = 0

    def _on_training_start(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "timesteps", "furthest_platform",
                 "total_platforms", "episode_reward"])

    def _on_step(self) -> bool:
        for done, info in zip(
                self.locals.get("dones", []),
                self.locals.get("infos", [])):
            if done:
                furthest  = info.get("furthest_platform", 0)
                total     = info.get("total_platforms",   40)
                ep_reward = info.get("episode", {}).get("r", float("nan"))
                self.episode_count += 1
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.episode_count, self.num_timesteps,
                        furthest, total, ep_reward,
                    ])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs",    type=int, default=8)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--resume",    action="store_true",
                        help="Resume training from platformer_model.zip")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # DummyVecEnv: runs envs sequentially in the same process.
    # This avoids the macOS multiprocessing 'spawn' bug that causes
    # SubprocVecEnv child processes to fail silently.
    print(f"Starting {args.n_envs} environments (DummyVecEnv)...")
    env_fns = [make_env(seed=args.seed + i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    lr_schedule = lr_schedule = get_linear_fn(3e-4, 1e-5, 1.0)

    if args.resume and os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            # rollout
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            # learning
            learning_rate=lr_schedule,
            gamma=0.99,
            gae_lambda=0.95,
            # exploration — high ent_coef prevents entropy collapse
            ent_coef=0.08,
            # clipping / value
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./tb_logs",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_platformer",
    )
    progress_cb = ProgressLoggerCallback(LOG_PATH)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, progress_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current model...")

    model.save(MODEL_PATH)
    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Progress log → {LOG_PATH}  (run plot_progress.py to visualise)")


if __name__ == "__main__":
    main()
