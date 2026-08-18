"""
Precision Platformer Gauntlet
------------------------------
A deterministic, exponentially-escalating jump gauntlet built for
reinforcement learning.

The goal is for the PPO agent to reach platform 40.

Difficulty increases through:
    - progressively larger gaps
    - progressively narrower platforms
    - optional moving platforms

The environment keeps the original physics and 13-value observation space.

Observation space (Box, 13 floats):
    [0]  agent y position (normalized)
    [1]  agent x velocity (normalized)
    [2]  agent y velocity (normalized)
    [3]  on_ground flag (0/1)
    [4]  progress along level (0-1)
    [5]  next platform: rel x_start (agent -> platform start)
    [6]  next platform: rel x_end
    [7]  next platform: rel y
    [8]  next platform: width (normalized)
    [9]  platform after next: rel x_start
    [10] platform after next: rel x_end
    [11] platform after next: rel y
    [12] platform after next: width (normalized)

Action space (MultiDiscrete [3, 2]):
    horizontal:
        0 = none
        1 = left
        2 = right

    jump:
        0 = no
        1 = yes
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
# Physics / world constants
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = 960, 540

GRAVITY = 0.6
JUMP_VELOCITY = -13.0
MOVE_SPEED = 5.0
TERMINAL_VY = 18.0

AGENT_W, AGENT_H = 26, 34


# ---------------------------------------------------------------------------
# Jump-distance calculation
# ---------------------------------------------------------------------------

# Approximate total airborne time for a jump that starts and ends
# at approximately the same vertical height.
_FLIGHT_FRAMES = (2 * abs(JUMP_VELOCITY)) / GRAVITY

MAX_JUMP_DISTANCE = MOVE_SPEED * _FLIGHT_FRAMES


# ---------------------------------------------------------------------------
# Level difficulty
# ---------------------------------------------------------------------------

NUM_PLATFORMS = 40

BASE_GAP = 70.0

# Keep the maximum gap below the theoretical maximum horizontal
# distance of the agent.
MAX_GAP = 0.90 * MAX_JUMP_DISTANCE

GAP_GROWTH = 0.12


BASE_WIDTH = 150.0
MIN_WIDTH = 46.0
WIDTH_SHRINK = 0.10


PLATFORM_THICKNESS = 18

Y_VARIATION = 40


# ---------------------------------------------------------------------------
# Moving platforms
# ---------------------------------------------------------------------------

# IMPORTANT:
# Start with moving platforms disabled while PPO relearns the basic
# platform-to-platform skill.
#
# Once the agent can reliably reach platform 40 on static platforms,
# change this to True and retrain.
ENABLE_MOVING_PLATFORMS = False

MOVING_PLATFORM_START_INDEX = 14
MOVING_PLATFORM_AMPLITUDE = 30
MOVING_PLATFORM_SPEED = 0.03


# ---------------------------------------------------------------------------
# Episode limit
# ---------------------------------------------------------------------------

MAX_EPISODE_STEPS = 1800


# ---------------------------------------------------------------------------
# Difficulty functions
# ---------------------------------------------------------------------------

def _gap_for_index(i: int) -> float:
    """
    Exponential approach from BASE_GAP toward MAX_GAP.

    The gap increases gradually rather than jumping immediately to
    the hardest value.
    """
    return (
        MAX_GAP
        - (MAX_GAP - BASE_GAP)
        * math.exp(-GAP_GROWTH * i)
    )


def _width_for_index(i: int) -> float:
    """
    Exponential platform-width reduction.

    The width is never allowed to become smaller than MIN_WIDTH.
    """
    width = BASE_WIDTH * ((1.0 - WIDTH_SHRINK) ** i)

    return max(MIN_WIDTH, width)


# ---------------------------------------------------------------------------
# Platform
# ---------------------------------------------------------------------------

class Platform:
    __slots__ = (
        "x",
        "y",
        "width",
        "moving",
        "base_y",
        "phase",
    )

    def __init__(
        self,
        x,
        y,
        width,
        moving=False,
        phase=0.0,
    ):
        self.x = float(x)
        self.y = float(y)
        self.width = float(width)

        self.moving = bool(moving)

        self.base_y = float(y)
        self.phase = float(phase)

    def update(self, t):
        """Update vertical position of a moving platform."""
        if self.moving:
            self.y = (
                self.base_y
                + MOVING_PLATFORM_AMPLITUDE
                * math.sin(
                    MOVING_PLATFORM_SPEED * t
                    + self.phase
                )
            )

    @property
    def x_end(self):
        return self.x + self.width


# ---------------------------------------------------------------------------
# Level generation
# ---------------------------------------------------------------------------

def generate_level(seed: int):
    """
    Generate the deterministic 40-platform gauntlet.

    The same seed always produces the same level.
    """

    rng = np.random.default_rng(seed)

    platforms = []

    x = 0.0
    y = SCREEN_H - 80

    # ------------------------------------------------------------------
    # Starting platform
    # ------------------------------------------------------------------

    platforms.append(
        Platform(
            x=x,
            y=y,
            width=220,
        )
    )

    x += 220


    # ------------------------------------------------------------------
    # Generate platforms 1 -> 40
    # ------------------------------------------------------------------

    for i in range(NUM_PLATFORMS):

        gap = _gap_for_index(i)
        width = _width_for_index(i)

        x += gap

        y = float(
            np.clip(
                y + rng.uniform(
                    -Y_VARIATION,
                    Y_VARIATION,
                ),
                220,
                SCREEN_H - 60,
            )
        )

        # Moving platforms are deliberately disabled initially.
        moving = (
            ENABLE_MOVING_PLATFORMS
            and i >= MOVING_PLATFORM_START_INDEX
            and rng.random() < 0.35
        )

        phase = float(
            rng.uniform(
                0,
                math.tau,
            )
        )

        platforms.append(
            Platform(
                x=x,
                y=y,
                width=width,
                moving=moving,
                phase=phase,
            )
        )

        x += width

    return platforms


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PlatformerGauntletEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        render_mode=None,
        seed=0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.seed_value = seed


        # --------------------------------------------------------------
        # Observation space
        # --------------------------------------------------------------

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(13,),
            dtype=np.float32,
        )


        # --------------------------------------------------------------
        # Action space
        # --------------------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [3, 2]
        )


        # --------------------------------------------------------------
        # Pygame state
        # --------------------------------------------------------------

        self.screen = None
        self.clock = None
        self.font = None


        self._reset_state()


    # ------------------------------------------------------------------
    # Reset internal state
    # ------------------------------------------------------------------

    def _reset_state(self):

        self.platforms = generate_level(
            self.seed_value
        )

        self.t = 0
        self.steps = 0


        # --------------------------------------------------------------
        # Starting position
        # --------------------------------------------------------------

        start = self.platforms[0]

        self.agent_x = (
            start.x
            + start.width / 2
            - AGENT_W / 2
        )

        self.agent_y = (
            start.y
            - AGENT_H
        )


        # --------------------------------------------------------------
        # Physics state
        # --------------------------------------------------------------

        self.vx = 0.0
        self.vy = 0.0

        self.on_ground = True


        # --------------------------------------------------------------
        # Progress tracking
        # --------------------------------------------------------------

        self.current_platform_idx = 0
        self.furthest_platform_idx = 0

        self.furthest_x = self.agent_x


    # ------------------------------------------------------------------
    # Gym reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.seed_value = seed

        self._reset_state()

        return (
            self._get_obs(),
            {
                "furthest_platform":
                    self.furthest_platform_idx
            },
        )


    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):

        # --------------------------------------------------------------
        # Next platform
        # --------------------------------------------------------------

        p1_index = min(
            self.current_platform_idx + 1,
            len(self.platforms) - 1,
        )

        p1 = self.platforms[p1_index]


        # --------------------------------------------------------------
        # Platform after next
        # --------------------------------------------------------------

        p2_index = min(
            self.current_platform_idx + 2,
            len(self.platforms) - 1,
        )

        p2 = self.platforms[p2_index]


        # --------------------------------------------------------------
        # Level length
        # --------------------------------------------------------------

        level_length = (
            self.platforms[-1].x_end
        )


        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        obs = np.array(
            [
                # Agent vertical position
                (self.agent_y / SCREEN_H) * 2.0 - 1.0,

                # Horizontal velocity
                np.clip(
                    self.vx / MOVE_SPEED,
                    -1.0,
                    1.0,
                ),

                # Vertical velocity
                np.clip(
                    self.vy / TERMINAL_VY,
                    -1.0,
                    1.0,
                ),

                # Ground state
                1.0 if self.on_ground else 0.0,

                # Overall level progress
                np.clip(
                    self.agent_x / level_length,
                    0.0,
                    1.0,
                ),

                # Next platform start
                np.clip(
                    (p1.x - self.agent_x)
                    / MAX_JUMP_DISTANCE,
                    -1.0,
                    1.0,
                ),

                # Next platform end
                np.clip(
                    (p1.x_end - self.agent_x)
                    / MAX_JUMP_DISTANCE,
                    -1.0,
                    1.0,
                ),

                # Next platform vertical position
                np.clip(
                    (p1.y - self.agent_y)
                    / SCREEN_H,
                    -1.0,
                    1.0,
                ),

                # Next platform width
                np.clip(
                    p1.width / BASE_WIDTH,
                    0.0,
                    1.0,
                ),

                # Second platform start
                np.clip(
                    (p2.x - self.agent_x)
                    / (2.0 * MAX_JUMP_DISTANCE),
                    -1.0,
                    1.0,
                ),

                # Second platform end
                np.clip(
                    (p2.x_end - self.agent_x)
                    / (2.0 * MAX_JUMP_DISTANCE),
                    -1.0,
                    1.0,
                ),

                # Second platform vertical position
                np.clip(
                    (p2.y - self.agent_y)
                    / SCREEN_H,
                    -1.0,
                    1.0,
                ),

                # Second platform width
                np.clip(
                    p2.width / BASE_WIDTH,
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

        return obs


    # ------------------------------------------------------------------
    # Environment step
    # ------------------------------------------------------------------

    def step(self, action):

        horiz = int(action[0])
        jump = int(action[1])


        # --------------------------------------------------------------
        # Advance simulation
        # --------------------------------------------------------------

        self.t += 1
        self.steps += 1


        # --------------------------------------------------------------
        # Update platforms
        # --------------------------------------------------------------

        for platform in self.platforms:
            platform.update(self.t)



        # --------------------------------------------------------------
        # Horizontal movement
        # --------------------------------------------------------------

        ACCELERATION = 2.0
        DECELERATION = 2.0

        if horiz == 1:
            self.vx = max(
                self.vx - ACCELERATION,
                -MOVE_SPEED,
            )

        elif horiz == 2:
            self.vx = min(
                self.vx + ACCELERATION,
                MOVE_SPEED,
            )

        else:
            if self.vx > 0:
                self.vx = max(
                    0.0,
                    self.vx - DECELERATION,
                )
            elif self.vx < 0:
                self.vx = min(
                    0.0,
                    self.vx + DECELERATION,
                )


        # --------------------------------------------------------------
        # Jump
        # --------------------------------------------------------------

        if jump == 1 and self.on_ground:

            self.vy = JUMP_VELOCITY

            self.on_ground = False


        # --------------------------------------------------------------
        # Gravity
        # --------------------------------------------------------------

        self.vy = min(
            self.vy + GRAVITY,
            TERMINAL_VY,
        )


        # --------------------------------------------------------------
        # Move agent
        # --------------------------------------------------------------

        previous_x = self.agent_x
        previous_y = self.agent_y

        self.agent_x += self.vx
        self.agent_y += self.vy


        # --------------------------------------------------------------
        # Base reward
        # --------------------------------------------------------------

        reward = -0.01

        terminated = False
        truncated = False


        # --------------------------------------------------------------
        # Small horizontal progress reward
        #
        # This remains intentionally small.
        #
        # We do NOT want PPO to learn:
        #
        #     "Just run right."
        #
        # The large reward should come from successfully
        # reaching new platforms.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Landing detection
        #
        # The agent may land on:
        #   - the platform it is currently on
        #   - the NEXT platform
        #
        # We use AABB horizontal overlap and a swept vertical check
        # so the agent cannot phase through a platform between frames.
        # --------------------------------------------------------------

        landed = False

        if self.vy >= 0:

            feet_y = self.agent_y + AGENT_H
            previous_feet_y = previous_y + AGENT_H

            # Check current platform and next platform.
            first_idx = max(
                0,
                self.current_platform_idx
            )

            last_idx = min(
                self.current_platform_idx + 1,
                len(self.platforms) - 1
            )

            for idx in range(first_idx, last_idx + 1):

                platform = self.platforms[idx]

                # --------------------------------------------------
                # Horizontal AABB overlap
                # --------------------------------------------------

                horizontal_overlap = (
                        self.agent_x + AGENT_W
                        > platform.x
                        and
                        self.agent_x
                        < platform.x_end
                )

                if not horizontal_overlap:
                    continue

                # --------------------------------------------------
                # Vertical crossing
                #
                # The agent must have been above the platform top
                # on the previous frame and now be at/below it.
                #
                # The extra <= check also handles the case where
                # the agent starts the frame already overlapping
                # the platform surface.
                # --------------------------------------------------

                crossed_platform_top = (
                        previous_feet_y <= platform.y
                        and
                        feet_y >= platform.y
                )

                if not crossed_platform_top:
                    continue

                # --------------------------------------------------
                # Successful landing
                # --------------------------------------------------

                self.agent_y = (
                        platform.y - AGENT_H
                )

                self.vy = 0.0
                self.on_ground = True

                landed = True

                # --------------------------------------------------
                # Only update progress when reaching a new platform
                # --------------------------------------------------

                if idx > self.furthest_platform_idx:
                    reward += 8.0

                    self.furthest_platform_idx = idx

                    self.furthest_x = max(
                        self.furthest_x,
                        self.agent_x,
                    )

                self.current_platform_idx = max(
                    self.current_platform_idx,
                    idx,
                )

                break


        # --------------------------------------------------------------
        # If we did not land, the agent is airborne.
        # --------------------------------------------------------------

        if not landed:
            self.on_ground = False


        # --------------------------------------------------------------
        # Fell into the void
        # --------------------------------------------------------------

        if (
            self.agent_y
            > SCREEN_H + 100
        ):

            reward -= 10.0

            terminated = True


        # --------------------------------------------------------------
        # Reached platform 40
        # --------------------------------------------------------------

        if (
            self.current_platform_idx
            == len(self.platforms) - 1
            and self.on_ground
        ):

            reward += 50.0

            terminated = True


        # --------------------------------------------------------------
        # Maximum episode length
        # --------------------------------------------------------------

        if self.steps >= MAX_EPISODE_STEPS:
            truncated = True


        # --------------------------------------------------------------
        # Information for callbacks / logging
        # --------------------------------------------------------------

        info = {
            "furthest_platform":
                self.furthest_platform_idx,

            "total_platforms":
                len(self.platforms) - 1,
        }


        # --------------------------------------------------------------
        # Optional rendering
        # --------------------------------------------------------------

        if self.render_mode == "human":
            self.render()


        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            info,
        )


    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):

        if pygame is None:
            raise RuntimeError(
                "pygame is required for rendering"
            )


        if self.screen is None:

            pygame.init()

            if self.render_mode == "human":

                self.screen = (
                    pygame.display.set_mode(
                        (SCREEN_W, SCREEN_H)
                    )
                )

                pygame.display.set_caption(
                    "Precision Platformer Gauntlet"
                )

            else:

                self.screen = pygame.Surface(
                    (SCREEN_W, SCREEN_H)
                )

            self.clock = pygame.time.Clock()

            self.font = pygame.font.SysFont(
                "consolas",
                20,
            )


        # --------------------------------------------------------------
        # Camera
        # --------------------------------------------------------------

        cam_x = (
            self.agent_x
            - SCREEN_W * 0.3
        )


        # --------------------------------------------------------------
        # Background
        # --------------------------------------------------------------

        self.screen.fill(
            (18, 18, 28)
        )


        # --------------------------------------------------------------
        # Platforms
        # --------------------------------------------------------------

        for idx, platform in enumerate(
            self.platforms
        ):

            sx = (
                platform.x
                - cam_x
            )

            if (
                sx + platform.width < 0
                or sx > SCREEN_W
            ):
                continue


            if (
                idx
                <= self.furthest_platform_idx
            ):

                color = (
                    90,
                    200,
                    120,
                )

            else:

                color = (
                    90,
                    110,
                    200,
                )


            pygame.draw.rect(
                self.screen,
                color,
                (
                    sx,
                    platform.y,
                    platform.width,
                    PLATFORM_THICKNESS,
                ),
            )


        # --------------------------------------------------------------
        # Agent
        # --------------------------------------------------------------

        ax = (
            self.agent_x
            - cam_x
        )

        pygame.draw.rect(
            self.screen,
            (240, 200, 60),
            (
                ax,
                self.agent_y,
                AGENT_W,
                AGENT_H,
            ),
        )


        # --------------------------------------------------------------
        # HUD
        # --------------------------------------------------------------

        hud = self.font.render(
            (
                f"Platform "
                f"{self.furthest_platform_idx}"
                f"/{len(self.platforms) - 1}"
                f"   Step {self.steps}"
            ),
            True,
            (230, 230, 230),
        )

        self.screen.blit(
            hud,
            (10, 10),
        )


        # --------------------------------------------------------------
        # Display
        # --------------------------------------------------------------

        if self.render_mode == "human":

            pygame.display.flip()

            self.clock.tick(
                self.metadata["render_fps"]
            )

        else:

            return np.transpose(
                pygame.surfarray.array3d(
                    self.screen
                ),
                (1, 0, 2),
            )


    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self):

        if (
            self.screen is not None
            and pygame is not None
        ):

            pygame.quit()

            self.screen = None