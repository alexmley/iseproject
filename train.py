"""
Train a PPO agent on the Precision Platformer Gauntlet.

Key hyperparameter changes vs v1:
  - ent_coef 0.005 → 0.08   (much more exploration forced)
  - n_steps  1024  → 2048   (longer rollouts to see jump consequences)
  - gamma    0.995 → 0.99
  - learning_rate fixed → linear decay 3e-4 → 1e-5
"""

import argparse
import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
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
    def __init__(self, log_path):
        super().__init__()
        self.log_path    = log_path
        self.episode_count = 0

    def _on_training_start(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "timesteps", "furthest_platform",
                 "total_platforms", "episode_reward"])

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info      = self.locals["infos"][i]
                furthest  = info.get("furthest_platform", 0)
                total     = info.get("total_platforms", 1)
                ep_reward = info.get("episode", {}).get("r", float("nan"))
                self.episode_count += 1
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [self.episode_count, self.num_timesteps,
                         furthest, total, ep_reward])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs",    type=int, default=8)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env_fns = [make_env(seed=args.seed + i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # Learning rate decays linearly from 3e-4 → 1e-5 over the full run
    lr_schedule = get_linear_fn(3e-4, 1e-5, 1.0)

    if args.resume and os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            # --- rollout ---
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            # --- learning ---
            learning_rate=lr_schedule,
            gamma=0.99,
            gae_lambda=0.95,
            # --- exploration: high ent_coef forces the agent to keep trying
            #     different actions instead of collapsing to "stand still"  ---
            ent_coef=0.08,
            # --- clipping ---
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

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, progress_cb],
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    print(f"\nSaved final model to {MODEL_PATH}")
    print(f"Progress log → {LOG_PATH}  (run plot_progress.py to visualise)")


if __name__ == "__main__":
    main()