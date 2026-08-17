"""
Watch a trained agent play the gauntlet in a live pygame window.

Usage:
    python play.py --model platformer_model.zip
    python play.py --model checkpoints/ppo_platformer_500000_steps.zip --seed 3
"""

import argparse
import time

from stable_baselines3 import PPO
from platformer_env import PlatformerGauntletEnv

import os
os.environ["SDL_VIDEODRIVER"] = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="platformer_model.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    env = PlatformerGauntletEnv(render_mode="human", seed=args.seed)
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(
            f"Episode {ep + 1}: reached platform "
            f"{info['furthest_platform']}/{info['total_platforms']}, "
            f"reward={total_reward:.2f}"
        )
        time.sleep(1.0)

    env.close()


if __name__ == "__main__":
    main()
