"""
Train PPO on the Precision Platformer Gauntlet.

Usage:
    python train.py --timesteps 3000000 --n-envs 8
    python train.py --resume
"""

import argparse, csv, os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from platformer_env import PlatformerGauntletEnv

LOG_PATH = "training_log.csv"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = "platformer_model.zip"


def make_env(seed):
    def _init():
        return Monitor(PlatformerGauntletEnv(render_mode=None, seed=seed))
    return _init


class ProgressLoggerCallback(BaseCallback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.episode_count = 0

    def _on_training_start(self):
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "timesteps", "furthest_platform",
                                    "total_platforms", "episode_reward"])

    def _on_step(self):
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done:
                self.episode_count += 1
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.episode_count, self.num_timesteps,
                        info.get("furthest_platform", 0),
                        info.get("total_platforms", 40),
                        info.get("episode", {}).get("r", float("nan")),
                    ])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])

    if args.resume and os.path.exists(MODEL_PATH):
        print(f"Resuming from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy", vec_env, verbose=1,
            n_steps=2048, batch_size=256, n_epochs=10,
            learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
            gamma=0.99, gae_lambda=0.95,
            ent_coef=0.04,
            clip_range=0.2, vf_coef=0.5, max_grad_norm=0.5,
            tensorboard_log="./tb_logs",
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[
                CheckpointCallback(
                    save_freq=max(100_000 // args.n_envs, 1),
                    save_path=CHECKPOINT_DIR,
                    name_prefix="ppo_platformer"),
                ProgressLoggerCallback(LOG_PATH),
            ],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving...")

    model.save(MODEL_PATH)
    print(f"Saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
