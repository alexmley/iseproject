"""
Plot the agent's improvement over training.

Reads training_log.csv (written by train.py) and shows:
  1. Furthest platform reached per episode, with a rolling average
  2. Episode reward over time

Usage:
    python plot_progress.py
    python plot_progress.py --log training_log.csv --window 100
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    episodes, furthest, total, rewards = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            furthest.append(int(row["furthest_platform"]))
            total.append(int(row["total_platforms"]))
            try:
                rewards.append(float(row["episode_reward"]))
            except ValueError:
                rewards.append(np.nan)
    return np.array(episodes), np.array(furthest), np.array(total), np.array(rewards)


def rolling_mean(x, window):
    if len(x) < window:
        return x, np.arange(1, len(x) + 1)
    kernel = np.ones(window) / window
    smoothed = np.convolve(x, kernel, mode="valid")
    x_axis = np.arange(window, len(x) + 1)
    return smoothed, x_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="training_log.csv")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    episodes, furthest, total, rewards = load_log(args.log)
    max_platforms = total[0] if len(total) else 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.scatter(episodes, furthest, s=4, alpha=0.25, color="#5a78c8", label="per episode")
    smooth, x_axis = rolling_mean(furthest, args.window)
    ax1.plot(x_axis, smooth, color="#e0a030", linewidth=2, label=f"{args.window}-episode rolling avg")
    ax1.axhline(max_platforms, color="green", linestyle="--", alpha=0.5, label="full level")
    ax1.set_ylabel("Furthest platform reached")
    ax1.set_title("Skill improvement over training")
    ax1.legend(loc="lower right")

    ax2.scatter(episodes, rewards, s=4, alpha=0.2, color="#c85a5a")
    smooth_r, x_axis_r = rolling_mean(np.nan_to_num(rewards), args.window)
    ax2.plot(x_axis_r, smooth_r, color="#5ac878", linewidth=2, label=f"{args.window}-episode rolling avg")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode reward")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_path = "progress.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
