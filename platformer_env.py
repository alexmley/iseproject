import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None

SCREEN_W, SCREEN_H = 960, 540
GRAVITY = 0.6
JUMP_VELOCITY = -13.0
MOVE_SPEED = 5.0
TERMINAL_VY = 18.0
AGENT_W, AGENT_H = 26, 34

_FLIGHT_FRAMES = (2 * abs(JUMP_VELOCITY)) / GRAVITY
MAX_JUMP_DISTANCE = MOVE_SPEED * _FLIGHT_FRAMES  # ~217px

NUM_PLATFORMS = 40
BASE_GAP = 70.0
MAX_GAP = 0.90 * MAX_JUMP_DISTANCE
GAP_GROWTH = 0.12
BASE_WIDTH = 150.0
MIN_WIDTH = 46.0
WIDTH_SHRINK = 0.10
PLATFORM_THICKNESS = 18
Y_VARIATION = 40
ENABLE_MOVING_PLATFORMS = False
MOVING_PLATFORM_START_INDEX = 14
MOVING_PLATFORM_AMPLITUDE = 30
MOVING_PLATFORM_SPEED = 0.03
MAX_EPISODE_STEPS = 1800


def _gap_for_index(i):
    raw = MAX_GAP - (MAX_GAP - BASE_GAP) * math.exp(-GAP_GROWTH * i)
    # curriculum: first 6 gaps are small so agent discovers landing
    if i < 6:
        return raw * 0.30
    return raw


def _width_for_index(i):
    return max(MIN_WIDTH, BASE_WIDTH * ((1.0 - WIDTH_SHRINK) ** i))


class Platform:
    __slots__ = ("x", "y", "width", "moving", "base_y", "phase")

    def __init__(self, x, y, width, moving=False, phase=0.0):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)
        self.moving = bool(moving)
        self.base_y = float(y)
        self.phase = float(phase)

    def update(self, t):
        if self.moving:
            self.y = self.base_y + MOVING_PLATFORM_AMPLITUDE * math.sin(
                MOVING_PLATFORM_SPEED * t + self.phase)

    @property
    def x_end(self):
        return self.x + self.width


def generate_level(seed):
    rng = np.random.default_rng(seed)
    platforms = []
    x, y = 0.0, float(SCREEN_H - 80)
    platforms.append(Platform(x, y, 220))
    x += 220
    for i in range(NUM_PLATFORMS):
        gap = _gap_for_index(i)
        width = _width_for_index(i)
        x += gap
        y = float(np.clip(y + rng.uniform(-Y_VARIATION, Y_VARIATION), 220, SCREEN_H - 60))
        moving = (ENABLE_MOVING_PLATFORMS and i >= MOVING_PLATFORM_START_INDEX
                  and rng.random() < 0.35)
        phase = float(rng.uniform(0, math.tau))
        platforms.append(Platform(x, y, width, moving=moving, phase=phase))
        x += width
    return platforms


class PlatformerGauntletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=0):
        super().__init__()
        self.render_mode = render_mode
        self.seed_value = seed
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 2])
        self.screen = self.clock = self.font = None
        self._reset_state()

    def _reset_state(self):
        self.platforms = generate_level(self.seed_value)
        self.t = self.steps = 0
        start = self.platforms[0]
        self.agent_x = start.x + start.width / 2 - AGENT_W / 2
        self.agent_y = start.y - AGENT_H
        self.vx = self.vy = 0.0
        self.on_ground = True
        self.current_platform_idx = 0
        self.furthest_platform_idx = 0
        self.furthest_x = self.agent_x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        self._reset_state()
        return self._get_obs(), {"furthest_platform": 0}

    def _get_obs(self):
        p1 = self.platforms[min(self.current_platform_idx + 1, len(self.platforms) - 1)]
        p2 = self.platforms[min(self.current_platform_idx + 2, len(self.platforms) - 1)]
        level_length = self.platforms[-1].x_end
        return np.array([
            (self.agent_y / SCREEN_H) * 2.0 - 1.0,
            np.clip(self.vx / MOVE_SPEED, -1.0, 1.0),
            np.clip(self.vy / TERMINAL_VY, -1.0, 1.0),
            1.0 if self.on_ground else 0.0,
            np.clip(self.agent_x / level_length, 0.0, 1.0),
            np.clip((p1.x - self.agent_x) / MAX_JUMP_DISTANCE, -1.0, 1.0),
            np.clip((p1.x_end - self.agent_x) / MAX_JUMP_DISTANCE, -1.0, 1.0),
            np.clip((p1.y - self.agent_y) / SCREEN_H, -1.0, 1.0),
            np.clip(p1.width / BASE_WIDTH, 0.0, 1.0),
            np.clip((p2.x - self.agent_x) / (2.0 * MAX_JUMP_DISTANCE), -1.0, 1.0),
            np.clip((p2.x_end - self.agent_x) / (2.0 * MAX_JUMP_DISTANCE), -1.0, 1.0),
            np.clip((p2.y - self.agent_y) / SCREEN_H, -1.0, 1.0),
            np.clip(p2.width / BASE_WIDTH, 0.0, 1.0),
        ], dtype=np.float32)

    def step(self, action):
        horiz, jump = int(action[0]), int(action[1])
        self.t += 1
        self.steps += 1

        for p in self.platforms:
            p.update(self.t)

        # Instant velocity (no acceleration) — cleaner learning signal
        if horiz == 1:
            self.vx = -MOVE_SPEED
        elif horiz == 2:
            self.vx = MOVE_SPEED
        else:
            self.vx = 0.0

        if jump == 1 and self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False

        self.vy = min(self.vy + GRAVITY, TERMINAL_VY)
        prev_ay = self.agent_y
        self.agent_x += self.vx
        self.agent_y += self.vy

        reward = -0.02
        terminated = False
        truncated = False

        # Progress reward only while grounded (prevents "run-and-fall" exploit)
        if self.on_ground and self.agent_x > self.furthest_x:
            reward += 0.03 * (self.agent_x - self.furthest_x)
            self.furthest_x = self.agent_x

        # AABB collision — scan ALL platforms ahead of current, not just +1
        # This prevents getting stuck if agent skips a platform index
        landed = False
        if self.vy >= 0:
            feet_y = self.agent_y + AGENT_H
            prev_feet_y = prev_ay + AGENT_H
            agent_left = self.agent_x
            agent_right = self.agent_x + AGENT_W

            # Scan from current platform up to current+3 to catch any skips
            scan_start = max(0, self.current_platform_idx)
            scan_end = min(self.current_platform_idx + 4, len(self.platforms))

            for idx in range(scan_start, scan_end):
                p = self.platforms[idx]
                if agent_right > p.x and agent_left < p.x_end:
                    if feet_y >= p.y and prev_feet_y <= p.y:
                        self.agent_y = p.y - AGENT_H
                        self.vy = 0.0
                        self.on_ground = True
                        self.current_platform_idx = max(self.current_platform_idx, idx)
                        landed = True
                        if idx > self.furthest_platform_idx:
                            # Strong, scaling bonus — clear signal for PPO
                            reward += 10.0 + 15.0 * (idx / NUM_PLATFORMS)
                            self.furthest_platform_idx = idx
                            self.furthest_x = max(self.furthest_x, self.agent_x)
                        break

        if not landed:
            self.on_ground = False

        if self.agent_y > SCREEN_H + 100:
            reward -= 5.0
            terminated = True

        if self.current_platform_idx == len(self.platforms) - 1 and self.on_ground:
            reward += 200.0
            terminated = True

        if self.steps >= MAX_EPISODE_STEPS:
            truncated = True

        info = {
            "furthest_platform": self.furthest_platform_idx,
            "total_platforms": len(self.platforms) - 1,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame required")
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("Precision Platformer Gauntlet")
            else:
                self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("consolas", 20)

        cam_x = self.agent_x - SCREEN_W * 0.3
        self.screen.fill((18, 18, 28))

        for idx, p in enumerate(self.platforms):
            sx = p.x - cam_x
            if sx + p.width < 0 or sx > SCREEN_W:
                continue
            color = (90, 200, 120) if idx <= self.furthest_platform_idx else (90, 110, 200)
            pygame.draw.rect(self.screen, color, (sx, p.y, p.width, PLATFORM_THICKNESS))

        ax = self.agent_x - cam_x
        pygame.draw.rect(self.screen, (240, 200, 60), (ax, self.agent_y, AGENT_W, AGENT_H))

        hud = self.font.render(
            f"Platform {self.furthest_platform_idx}/{len(self.platforms) - 1}   Step {self.steps}",
            True, (230, 230, 230))
        self.screen.blit(hud, (10, 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.quit()
            self.screen = None
