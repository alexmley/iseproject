"""
Precision Platformer Gauntlet  –  v3 (fixed)
----------------------------------------------
Key fixes vs v1:
  - No jump cooldown: was causing the agent to walk off platforms before
    it could jump. PPO learns not to jump immediately on its own.
  - Curriculum warmup: first 8 gaps are 55% of their normal size so the
    agent can discover "landing" before the exponential curve bites hard.
  - Wider starting platforms and gentler initial gap/shrink constants.
  - Observation is 13 floats (same structure as v1, cooldown removed).
  - Landing bonus scales with platform index so deeper landings feel more
    rewarding even early in training.

Observation space (Box, 13 floats):
    [0]  agent y position (normalised 0-1, flipped so up=positive)
    [1]  agent x velocity (normalised)
    [2]  agent y velocity (normalised)
    [3]  on_ground flag (0/1)
    [4]  progress along level (0-1)
    [5]  next platform: rel x_start
    [6]  next platform: rel x_end
    [7]  next platform: rel y
    [8]  next platform: width (normalised)
    [9]  platform after next: rel x_start
    [10] platform after next: rel x_end
    [11] platform after next: rel y
    [12] platform after next: width (normalised)

Action space (MultiDiscrete [3, 2]):
    horizontal: 0=none  1=left  2=right
    jump:       0=no    1=yes  (only fires if on_ground)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 960, 540
GRAVITY       = 0.6
JUMP_VELOCITY = -13.0
MOVE_SPEED    = 5.0
TERMINAL_VY   = 18.0
AGENT_W, AGENT_H = 26, 34

_FLIGHT_FRAMES    = (2 * abs(JUMP_VELOCITY)) / GRAVITY
MAX_JUMP_DISTANCE = MOVE_SPEED * _FLIGHT_FRAMES      # ≈ 217 px

# ---------------------------------------------------------------------------
# Level / difficulty constants
# ---------------------------------------------------------------------------
NUM_PLATFORMS = 40

BASE_GAP      = 55.0
MAX_GAP       = 0.88 * MAX_JUMP_DISTANCE             # ≈ 191 px
GAP_GROWTH    = 0.10

BASE_WIDTH    = 160.0
MIN_WIDTH     = 46.0
WIDTH_SHRINK  = 0.09

PLATFORM_THICKNESS         = 18
Y_VARIATION                = 36
MOVING_PLATFORM_START      = 16
MOVING_PLATFORM_AMPLITUDE  = 28
MOVING_PLATFORM_SPEED      = 0.03

MAX_EPISODE_STEPS = 2000


def _gap_for_index(i: int) -> float:
    """Exponential ramp BASE_GAP → MAX_GAP.
       First 8 platforms use 55 % of the normal gap (curriculum warmup)."""
    raw = MAX_GAP - (MAX_GAP - BASE_GAP) * math.exp(-GAP_GROWTH * i)
    return raw * 0.55 if i < 8 else raw


def _width_for_index(i: int) -> float:
    return max(MIN_WIDTH, BASE_WIDTH * ((1 - WIDTH_SHRINK) ** i))


class Platform:
    __slots__ = ("x", "y", "width", "moving", "base_y", "phase")

    def __init__(self, x, y, width, moving=False, phase=0.0):
        self.x, self.y, self.width = x, y, width
        self.moving, self.base_y, self.phase = moving, y, phase

    def update(self, t):
        if self.moving:
            self.y = self.base_y + MOVING_PLATFORM_AMPLITUDE * math.sin(
                MOVING_PLATFORM_SPEED * t + self.phase)

    @property
    def x_end(self):
        return self.x + self.width


def generate_level(seed: int):
    rng = np.random.default_rng(seed)
    platforms = []
    x, y = 0.0, float(SCREEN_H - 80)
    platforms.append(Platform(x, y, 240))   # wide starting pad
    x += 240

    for i in range(NUM_PLATFORMS):
        gap   = _gap_for_index(i)
        width = _width_for_index(i)
        x    += gap
        y     = float(np.clip(
            y + rng.uniform(-Y_VARIATION, Y_VARIATION), 200, SCREEN_H - 60))
        moving = i >= MOVING_PLATFORM_START and rng.random() < 0.35
        phase  = float(rng.uniform(0, math.tau))
        platforms.append(Platform(x, y, width, moving=moving, phase=phase))
        x += width

    return platforms


class PlatformerGauntletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, seed=0):
        super().__init__()
        self.render_mode = render_mode
        self.seed_value  = seed

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
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
        self.on_ground             = True
        self.current_platform_idx  = 0
        self.furthest_platform_idx = 0
        self.furthest_x            = self.agent_x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed
        self._reset_state()
        return self._get_obs(), {"furthest_platform": 0}

    def _get_obs(self):
        p1 = self.platforms[min(self.current_platform_idx + 1,
                                len(self.platforms) - 1)]
        p2 = self.platforms[min(self.current_platform_idx + 2,
                                len(self.platforms) - 1)]
        level_length = self.platforms[-1].x_end

        return np.array([
            (self.agent_y / SCREEN_H) * 2 - 1,
            np.clip(self.vx / MOVE_SPEED,    -1, 1),
            np.clip(self.vy / TERMINAL_VY,   -1, 1),
            1.0 if self.on_ground else 0.0,
            np.clip(self.agent_x / level_length, 0, 1),
            np.clip((p1.x     - self.agent_x) / MAX_JUMP_DISTANCE,       -1, 1),
            np.clip((p1.x_end - self.agent_x) / MAX_JUMP_DISTANCE,       -1, 1),
            np.clip((p1.y     - self.agent_y) / SCREEN_H,                -1, 1),
            np.clip(p1.width  / BASE_WIDTH,                                0, 1),
            np.clip((p2.x     - self.agent_x) / (2*MAX_JUMP_DISTANCE),   -1, 1),
            np.clip((p2.x_end - self.agent_x) / (2*MAX_JUMP_DISTANCE),   -1, 1),
            np.clip((p2.y     - self.agent_y) / SCREEN_H,                -1, 1),
            np.clip(p2.width  / BASE_WIDTH,                                0, 1),
        ], dtype=np.float32)

    def step(self, action):
        horiz, jump = int(action[0]), int(action[1])
        self.t     += 1
        self.steps += 1

        for p in self.platforms:
            p.update(self.t)

        if   horiz == 1: self.vx = -MOVE_SPEED
        elif horiz == 2: self.vx =  MOVE_SPEED
        else:            self.vx =  0.0

        if jump == 1 and self.on_ground:
            self.vy       = JUMP_VELOCITY
            self.on_ground = False

        self.vy       = min(self.vy + GRAVITY, TERMINAL_VY)
        prev_ay       = self.agent_y
        self.agent_x += self.vx
        self.agent_y += self.vy

        reward     = -0.05
        terminated = False
        truncated  = False

        if self.agent_x > self.furthest_x:
            reward         += 0.05 * (self.agent_x - self.furthest_x)
            self.furthest_x = self.agent_x

        # Collision (only while falling: vy >= 0)
        landed = False
        if self.vy >= 0:
            feet_y      = self.agent_y + AGENT_H
            prev_feet_y = prev_ay + AGENT_H
            mid_x       = self.agent_x + AGENT_W * 0.5
            for idx, p in enumerate(self.platforms):
                if p.x <= mid_x <= p.x_end:
                    if feet_y >= p.y and prev_feet_y <= p.y:
                        self.agent_y              = p.y - AGENT_H
                        self.vy                   = 0.0
                        self.on_ground            = True
                        self.current_platform_idx = idx
                        landed                    = True
                        if idx > self.furthest_platform_idx:
                            reward += 2.0 + 5.0 * (idx / NUM_PLATFORMS)
                            self.furthest_platform_idx = idx
                        break

        if not landed:
            self.on_ground = False

        # tiny airborne bonus: makes jumping feel marginally better
        # than standing still, helping break entropy collapse
        if not self.on_ground and self.vy < 0:
            reward += 0.005

        if self.agent_y > SCREEN_H + 100:
            reward    -= 10.0
            terminated = True

        if (self.current_platform_idx == len(self.platforms) - 1
                and self.on_ground):
            reward    += 100.0
            terminated = True

        if self.steps >= MAX_EPISODE_STEPS:
            truncated = True

        info = {
            "furthest_platform": self.furthest_platform_idx,
            "total_platforms":   len(self.platforms) - 1,
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame is required for rendering")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
                pygame.display.set_caption("Precision Platformer Gauntlet")
            else:
                self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()
            self.font  = pygame.font.SysFont("consolas", 20)

        cam_x = self.agent_x - SCREEN_W * 0.3
        self.screen.fill((18, 18, 28))

        for idx, p in enumerate(self.platforms):
            sx = p.x - cam_x
            if sx + p.width < 0 or sx > SCREEN_W:
                continue
            color = (90, 200, 120) if idx <= self.furthest_platform_idx \
                    else (90, 110, 200)
            pygame.draw.rect(self.screen, color,
                             (sx, p.y, p.width, PLATFORM_THICKNESS))

        # agent
        ax = self.agent_x - cam_x
        pygame.draw.rect(self.screen, (240, 200, 60),
                         (ax, self.agent_y, AGENT_W, AGENT_H))

        hud = self.font.render(
            f"Platform {self.furthest_platform_idx}/{len(self.platforms)-1}"
            f"   Step {self.steps}", True, (230, 230, 230))
        self.screen.blit(hud, (10, 10))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.quit()
            self.screen = None
