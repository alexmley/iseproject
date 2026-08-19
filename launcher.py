"""
launcher.py — Precision Platformer Gauntlet

Modes:
    Watch AI
    Play
    Chart

Controls:
    Menu:
        ↑ / ↓ or W / S  - navigate
        ENTER / SPACE   - select

    AI Watch:
        1 - 1×
        2 - 2×
        3 - 5×
        4 - MAX
        - - slower
        + / = - faster
        ESC - menu

    Human:
        ← / → - move
        SPACE / ↑ - jump
        ESC - menu
"""

import os
import sys
import time
from collections import deque

import pygame
import numpy as np

from platformer_env import (
    PlatformerGauntletEnv,
    SCREEN_W,
    SCREEN_H,
    AGENT_W,
    AGENT_H,
    GRAVITY,
    JUMP_VELOCITY,
    MOVE_SPEED,
    TERMINAL_VY,
    PLATFORM_THICKNESS,
    generate_level,
)


# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

BG         = (12, 14, 23)
PANEL      = (20, 24, 40)
ACCENT     = (99, 210, 140)
PLATFORM_C = (75, 95, 180)
AGENT_COL  = (240, 200, 60)

TEXT_HI    = (230, 230, 230)
TEXT_MID   = (150, 155, 170)
TEXT_LO    = (110, 120, 150)

DANGER     = (220, 80, 80)
GOLD       = (255, 200, 60)


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

FPS = 60

SPEEDS = [1, 2, 5, 0]
SPEED_LBLS = ["1×", "2×", "5×", "MAX"]

# Length of the snake trail.
# Increase this for a longer trail, decrease for a shorter one.
TRAIL_LENGTH = 22

# Maximum visible trail thickness.
TRAIL_WIDTH = 3


# ─────────────────────────────────────────────────────────────────────────────
# MODEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def list_checkpoints():
    items = []
    ckpt = "checkpoints"

    if os.path.isdir(ckpt):
        zips = sorted(
            [f for f in os.listdir(ckpt) if f.endswith(".zip")],
            key=lambda f: int("".join(filter(str.isdigit, f)) or 0),
        )

        for z in zips:
            steps = int("".join(filter(str.isdigit, z)) or 0)
            items.append(
                (
                    f"{steps:,} steps",
                    os.path.join(ckpt, z),
                )
            )

    if os.path.exists("platformer_model.zip"):
        items.append(("Final model", "platformer_model.zip"))

    return items


def load_model(path):
    try:
        from stable_baselines3 import PPO
        return PPO.load(path)
    except Exception as e:
        print(f"Load error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# COLLISION
# ─────────────────────────────────────────────────────────────────────────────

def check_landing(ax, ay, prev_ay, vy, platforms, cur_idx):
    if vy < 0:
        return False, ay, vy, cur_idx

    feet_y = ay + AGENT_H
    prev_feet_y = prev_ay + AGENT_H

    ar = ax + AGENT_W

    scan_end = min(cur_idx + 4, len(platforms))

    for idx in range(max(0, cur_idx), scan_end):
        p = platforms[idx]

        if ar > p.x and ax < p.x_end:
            if feet_y >= p.y and prev_feet_y <= p.y:
                return (
                    True,
                    p.y - AGENT_H,
                    0.0,
                    idx,
                )

    return False, ay, vy, cur_idx


# ─────────────────────────────────────────────────────────────────────────────
# TRAIL
# ─────────────────────────────────────────────────────────────────────────────

def draw_trail(surface, trail, cam_x):
    """
    Draw a small green snake-like trail behind the agent.

    The trail stores world-space positions, then converts them to
    screen-space using the current camera position.

    Older points are thinner and darker.
    Newer points become progressively brighter/thicker.
    """

    if len(trail) < 2:
        return

    points = [
        (int(x - cam_x), int(y))
        for x, y in trail
    ]

    total = len(points)

    for i in range(1, total):
        # 0 = old end, 1 = newest end
        progress = i / max(total - 1, 1)

        # Keep the trail subtle.
        width = max(
            1,
            int(1 + progress * (TRAIL_WIDTH - 1))
        )

        # Blend from dark green -> bright green.
        old_col = (35, 85, 65)
        new_col = ACCENT

        r = int(old_col[0] + (new_col[0] - old_col[0]) * progress)
        g = int(old_col[1] + (new_col[1] - old_col[1]) * progress)
        b = int(old_col[2] + (new_col[2] - old_col[2]) * progress)

        pygame.draw.line(
            surface,
            (r, g, b),
            points[i - 1],
            points[i],
            width,
        )


def add_trail_point(trail, x, y):
    """
    Add the agent's centre position to the trail.

    deque automatically removes the oldest point once the
    fixed maximum length is reached.
    """

    trail.append(
        (
            x + AGENT_W / 2,
            y + AGENT_H / 2,
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DRAW
# ─────────────────────────────────────────────────────────────────────────────

def draw_level(surface, platforms, cam_x, furthest_idx):
    for idx, p in enumerate(platforms):
        sx = p.x - cam_x

        if sx + p.width < 0 or sx > SCREEN_W:
            continue

        col = ACCENT if idx <= furthest_idx else PLATFORM_C

        pygame.draw.rect(
            surface,
            col,
            (
                int(sx),
                int(p.y),
                int(p.width),
                PLATFORM_THICKNESS,
            ),
        )


def draw_agent(surface, ax, ay):
    pygame.draw.rect(
        surface,
        AGENT_COL,
        (
            int(ax),
            int(ay),
            AGENT_W,
            AGENT_H,
        ),
    )


def draw_bar(surface, font, furthest, total):
    bw, bh = 260, 10
    bx, by = SCREEN_W - bw - 16, 14

    pct = furthest / max(total, 1)

    pygame.draw.rect(
        surface,
        PANEL,
        (bx, by, bw, bh),
        border_radius=4,
    )

    if pct > 0:
        pygame.draw.rect(
            surface,
            ACCENT,
            (
                bx,
                by,
                int(bw * pct),
                bh,
            ),
            border_radius=4,
        )

    pygame.draw.rect(
        surface,
        (50, 58, 90),
        (bx, by, bw, bh),
        1,
        border_radius=4,
    )

    lbl = font.render(
        f"Platform  {furthest} / {total}",
        True,
        TEXT_MID,
    )

    surface.blit(
        lbl,
        (
            bx + bw - lbl.get_width(),
            by + bh + 5,
        ),
    )


def draw_speed_bar(surface, font, speed_idx):
    """
    Minimal speed indicator.

    1 / 2 / 3 / 4 correspond to the four speed modes.
    """

    labels = [
        "1  1×",
        "2  2×",
        "3  5×",
        "4  MAX",
    ]

    x = 16
    y = 12

    for i, label in enumerate(labels):
        active = i == speed_idx

        text_col = ACCENT if active else TEXT_LO

        txt = font.render(
            label,
            True,
            text_col,
        )

        surface.blit(
            txt,
            (x, y),
        )

        x += txt.get_width() + 16

    # +/- hint
    hint = font.render(
        "− / + speed",
        True,
        TEXT_LO,
    )

    surface.blit(
        hint,
        (
            x + 4,
            y,
        ),
    )


def draw_hint(surface, font, text, y=16):
    surface.blit(
        font.render(text, True, TEXT_LO),
        (16, y),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MENU
# ─────────────────────────────────────────────────────────────────────────────

def run_menu(screen, fonts):
    font_title, font_body, font_sm = fonts

    clock = pygame.time.Clock()

    has_model = bool(list_checkpoints())
    has_log = os.path.exists("training_log.csv")

    options = [
        ("Watch AI", has_model, "ai"),
        ("Play", True, "human"),
        ("Training Chart", has_log, "chart"),
    ]

    sel = 0

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(options)

                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(options)

                elif event.key in (
                    pygame.K_RETURN,
                    pygame.K_SPACE,
                ):
                    if options[sel][1]:
                        return options[sel][2]

                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # ── background ──────────────────────────────────────────────────────

        screen.fill(BG)

        # ── title ───────────────────────────────────────────────────────────

        title = font_title.render(
            "PRECISION PLATFORMER",
            True,
            TEXT_HI,
        )

        screen.blit(
            title,
            (
                SCREEN_W // 2 - title.get_width() // 2,
                86,
            ),
        )

        sub = font_sm.render(
            "GAUNTLET",
            True,
            ACCENT,
        )

        screen.blit(
            sub,
            (
                SCREEN_W // 2 - sub.get_width() // 2,
                86 + title.get_height() + 8,
            ),
        )

        # Small divider
        divider_y = 155

        pygame.draw.line(
            screen,
            (35, 42, 65),
            (SCREEN_W // 2 - 170, divider_y),
            (SCREEN_W // 2 + 170, divider_y),
            1,
        )

        # ── menu options ────────────────────────────────────────────────────

        start_y = 210
        spacing = 62

        for i, (name, enabled, _) in enumerate(options):

            y = start_y + i * spacing

            is_selected = i == sel

            # Selection marker
            if is_selected:
                pygame.draw.rect(
                    screen,
                    ACCENT,
                    (
                        SCREEN_W // 2 - 145,
                        y + 5,
                        3,
                        30,
                    ),
                )

            col = (
                TEXT_HI
                if enabled
                else TEXT_LO
            )

            txt = font_body.render(
                name,
                True,
                col,
            )

            screen.blit(
                txt,
                (
                    SCREEN_W // 2 - txt.get_width() // 2,
                    y,
                ),
            )

            if not enabled:
                unavailable = font_sm.render(
                    "unavailable",
                    True,
                    DANGER,
                )

                screen.blit(
                    unavailable,
                    (
                        SCREEN_W // 2
                        + txt.get_width() // 2
                        + 15,
                        y + 5,
                    ),
                )

        # ── controls ────────────────────────────────────────────────────────

        controls = font_sm.render(
            "↑ ↓  navigate     ENTER  select     ESC  quit",
            True,
            TEXT_LO,
        )

        screen.blit(
            controls,
            (
                SCREEN_W // 2 - controls.get_width() // 2,
                SCREEN_H - 42,
            ),
        )

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT PICKER
# ─────────────────────────────────────────────────────────────────────────────

def run_checkpoint_picker(screen, fonts):
    font_title, font_body, font_sm = fonts

    clock = pygame.time.Clock()

    items = list_checkpoints()

    if not items:
        return None

    sel = len(items) - 1
    scroll = 0

    VIS = 8
    ih = 46

    while True:
        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key in (pygame.K_UP, pygame.K_w):
                    sel = max(0, sel - 1)
                    scroll = min(scroll, sel)

                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    sel = min(len(items) - 1, sel + 1)
                    scroll = max(
                        0,
                        sel - VIS + 1,
                    )

                elif event.key in (
                    pygame.K_RETURN,
                    pygame.K_SPACE,
                ):
                    return items[sel]

                elif event.key == pygame.K_ESCAPE:
                    return None

        screen.fill(BG)

        title = font_title.render(
            "SELECT CHECKPOINT",
            True,
            TEXT_HI,
        )

        screen.blit(
            title,
            (
                SCREEN_W // 2 - title.get_width() // 2,
                60,
            ),
        )

        sub = font_sm.render(
            "ENTER to load",
            True,
            TEXT_LO,
        )

        screen.blit(
            sub,
            (
                SCREEN_W // 2 - sub.get_width() // 2,
                60 + title.get_height() + 8,
            ),
        )

        iw = 480
        ix = SCREEN_W // 2 - iw // 2
        sy = 140

        visible = items[
            scroll:scroll + VIS
        ]

        for i, (label, path) in enumerate(visible):

            real_idx = scroll + i
            iy = sy + i * (ih + 7)

            is_sel = real_idx == sel
            is_final = path == "platformer_model.zip"

            if is_sel:
                pygame.draw.rect(
                    screen,
                    ACCENT,
                    (
                        ix,
                        iy + 7,
                        3,
                        ih - 14,
                    ),
                )

            col = (
                TEXT_HI
                if is_sel
                else TEXT_MID
            )

            ls = font_body.render(
                label,
                True,
                col,
            )

            screen.blit(
                ls,
                (
                    ix + 18,
                    iy + ih // 2
                    - ls.get_height() // 2,
                ),
            )

            if is_final:
                tag = font_sm.render(
                    "FINAL",
                    True,
                    GOLD,
                )

                screen.blit(
                    tag,
                    (
                        ix + iw - tag.get_width(),
                        iy + ih // 2
                        - tag.get_height() // 2,
                    ),
                )

        foot = font_sm.render(
            "↑ ↓ select     ENTER load     ESC back",
            True,
            TEXT_LO,
        )

        screen.blit(
            foot,
            (
                SCREEN_W // 2 - foot.get_width() // 2,
                SCREEN_H - 30,
            ),
        )

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# AI WATCH
# ─────────────────────────────────────────────────────────────────────────────

def run_ai_mode(screen, fonts):
    _, font_body, font_sm = fonts

    clock = pygame.time.Clock()

    chosen = run_checkpoint_picker(
        screen,
        fonts,
    )

    if chosen is None:
        return

    label, path = chosen

    screen.fill(BG)

    msg = font_body.render(
        f"Loading {label}...",
        True,
        TEXT_MID,
    )

    screen.blit(
        msg,
        (
            SCREEN_W // 2
            - msg.get_width() // 2,
            SCREEN_H // 2,
        ),
    )

    pygame.display.flip()

    model = load_model(path)

    if model is None:
        return

    env = PlatformerGauntletEnv(
        render_mode=None,
        seed=0,
    )

    episode = 0
    speed_idx = 0

    while True:

        episode += 1

        obs, _ = env.reset(seed=episode)

        done = False

        # New trail for each episode.
        trail = deque(
            maxlen=TRAIL_LENGTH
        )

        while not done:

            spd = SPEEDS[speed_idx]

            if spd:
                clock.tick(FPS * spd)
            else:
                pygame.event.pump()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                    # Direct speed selection
                    if event.key == pygame.K_1:
                        speed_idx = 0

                    elif event.key == pygame.K_2:
                        speed_idx = 1

                    elif event.key == pygame.K_3:
                        speed_idx = 2

                    elif event.key == pygame.K_4:
                        speed_idx = 3

                    # Slower
                    elif event.key in (
                        pygame.K_MINUS,
                        pygame.K_KP_MINUS,
                    ):
                        speed_idx = max(
                            0,
                            speed_idx - 1,
                        )

                    # Faster
                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        speed_idx = min(
                            len(SPEEDS) - 1,
                            speed_idx + 1,
                        )

            # ── PPO prediction ──────────────────────────────────────────────

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, _, terminated, truncated, _ = env.step(
                action
            )

            done = terminated or truncated

            # ── trail ───────────────────────────────────────────────────────

            add_trail_point(
                trail,
                env.agent_x,
                env.agent_y,
            )

            # ── rendering ──────────────────────────────────────────────────

            if spd <= 2 or episode % 3 == 0:

                cam_x = (
                    env.agent_x
                    - SCREEN_W * 0.3
                )

                screen.fill(BG)

                draw_level(
                    screen,
                    env.platforms,
                    cam_x,
                    env.furthest_platform_idx,
                )

                # Trail must be drawn BEFORE the agent.
                draw_trail(
                    screen,
                    trail,
                    cam_x,
                )

                draw_agent(
                    screen,
                    env.agent_x - cam_x,
                    env.agent_y,
                )

                draw_bar(
                    screen,
                    font_sm,
                    env.furthest_platform_idx,
                    len(env.platforms) - 1,
                )

                draw_speed_bar(
                    screen,
                    font_sm,
                    speed_idx,
                )

                draw_hint(
                    screen,
                    font_sm,
                    f"Episode {episode}   [{label}]   ESC → menu",
                    y=44,
                )

                pygame.display.flip()

        time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN PLAY
# ─────────────────────────────────────────────────────────────────────────────

def run_human_mode(screen, fonts):
    _, font_body, font_sm = fonts

    clock = pygame.time.Clock()

    best_ever = 0
    episode = 0

    while True:

        episode += 1

        platforms = generate_level(seed=0)

        start = platforms[0]

        ax = (
            start.x
            + start.width / 2
            - AGENT_W / 2
        )

        ay = start.y - AGENT_H

        vx = vy = 0.0

        prev_ay = ay

        on_ground = True

        furthest = 0
        cur_idx = 0

        t = 0

        # ── trail ───────────────────────────────────────────────────────────

        trail = deque(
            maxlen=TRAIL_LENGTH
        )

        running = True

        while running:

            clock.tick(FPS)

            t += 1

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    return

            keys = pygame.key.get_pressed()

            # ── movement ────────────────────────────────────────────────────

            vx = (
                -MOVE_SPEED
                if keys[pygame.K_LEFT]
                else MOVE_SPEED
                if keys[pygame.K_RIGHT]
                else 0.0
            )

            if (
                keys[pygame.K_SPACE]
                or keys[pygame.K_UP]
            ) and on_ground:

                vy = JUMP_VELOCITY
                on_ground = False

            # ── physics ─────────────────────────────────────────────────────

            vy = min(
                vy + GRAVITY,
                TERMINAL_VY,
            )

            prev_ay = ay

            ax += vx
            ay += vy

            for p in platforms:
                p.update(t)

            # ── collision ──────────────────────────────────────────────────

            landed, ay, vy, new_idx = check_landing(
                ax,
                ay,
                prev_ay,
                vy,
                platforms,
                cur_idx,
            )

            if landed:

                on_ground = True

                cur_idx = new_idx

                if new_idx > furthest:

                    furthest = new_idx

                    best_ever = max(
                        best_ever,
                        furthest,
                    )

            else:
                on_ground = False

            # ── trail ───────────────────────────────────────────────────────

            add_trail_point(
                trail,
                ax,
                ay,
            )

            # ── camera ──────────────────────────────────────────────────────

            cam_x = (
                ax
                - SCREEN_W * 0.3
            )

            screen.fill(BG)

            draw_level(
                screen,
                platforms,
                cam_x,
                furthest,
            )

            # Trail behind player
            draw_trail(
                screen,
                trail,
                cam_x,
            )

            draw_agent(
                screen,
                ax - cam_x,
                ay,
            )

            draw_bar(
                screen,
                font_sm,
                furthest,
                len(platforms) - 1,
            )

            draw_hint(
                screen,
                font_sm,
                f"← → move   SPACE jump   ESC menu   Best: {best_ever}",
            )

            pygame.display.flip()

            # ── death ───────────────────────────────────────────────────────

            if ay > SCREEN_H + 100:
                running = False

        # ── death screen ────────────────────────────────────────────────────

        overlay = pygame.Surface(
            (SCREEN_W, SCREEN_H),
            pygame.SRCALPHA,
        )

        overlay.fill(
            (0, 0, 0, 150)
        )

        screen.blit(
            overlay,
            (0, 0),
        )

        reached = font_body.render(
            f"Reached platform {furthest} / {len(platforms)-1}",
            True,
            TEXT_HI,
        )

        best = font_sm.render(
            f"Personal best: {best_ever}",
            True,
            TEXT_MID,
        )

        retry = font_sm.render(
            "ENTER retry     ESC menu",
            True,
            TEXT_LO,
        )

        screen.blit(
            reached,
            (
                SCREEN_W // 2
                - reached.get_width() // 2,
                SCREEN_H // 2 - 40,
            ),
        )

        screen.blit(
            best,
            (
                SCREEN_W // 2
                - best.get_width() // 2,
                SCREEN_H // 2,
            ),
        )

        screen.blit(
            retry,
            (
                SCREEN_W // 2
                - retry.get_width() // 2,
                SCREEN_H // 2 + 36,
            ),
        )

        pygame.display.flip()

        waiting = True

        while waiting:

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_RETURN:
                        waiting = False

                    elif event.key == pygame.K_ESCAPE:
                        return


# ─────────────────────────────────────────────────────────────────────────────
# CHART
# ─────────────────────────────────────────────────────────────────────────────

def run_chart_mode(screen, fonts):
    _, font_body, font_sm = fonts

    clock = pygame.time.Clock()

    import csv

    try:

        with open("training_log.csv") as f:
            rows = list(csv.DictReader(f))

        vals = [
            int(r["furthest_platform"])
            for r in rows
        ]

        total = (
            int(rows[0]["total_platforms"])
            if rows
            else 40
        )

    except Exception:

        vals = []
        total = 40

    if not vals:

        while True:

            screen.fill(BG)

            msg = font_body.render(
                "No training_log.csv — run train.py first",
                True,
                DANGER,
            )

            screen.blit(
                msg,
                (
                    SCREEN_W // 2
                    - msg.get_width() // 2,
                    SCREEN_H // 2,
                ),
            )

            pygame.display.flip()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                ):
                    return

            clock.tick(FPS)

        return

    window = max(
        1,
        len(vals) // 20,
    )

    smooth = np.convolve(
        vals,
        np.ones(window) / window,
        mode="valid",
    )

    PAD = 64

    cw = SCREEN_W - PAD * 2
    ch = SCREEN_H - PAD * 2 - 48

    def to_px(i, v, n):
        return (
            PAD
            + int(
                i / max(n - 1, 1)
                * cw
            ),
            PAD
            + int(
                (
                    1
                    - v / max(total, 1)
                )
                * ch
            ),
        )

    while True:

        clock.tick(FPS)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_ESCAPE
            ):
                return

        screen.fill(BG)

        for tick in range(
            0,
            total + 1,
            5,
        ):

            gy = (
                PAD
                + int(
                    (
                        1
                        - tick / total
                    )
                    * ch
                )
            )

            pygame.draw.line(
                screen,
                PANEL,
                (PAD, gy),
                (PAD + cw, gy),
            )

            screen.blit(
                font_sm.render(
                    str(tick),
                    True,
                    TEXT_LO,
                ),
                (
                    PAD - 30,
                    gy - 8,
                ),
            )

        n = len(vals)

        for i, v in enumerate(vals):

            pygame.draw.circle(
                screen,
                (*PLATFORM_C, 100),
                to_px(i, v, n),
                2,
            )

        off = (
            n - len(smooth)
        ) // 2

        pts = [
            to_px(
                off + i,
                v,
                n,
            )
            for i, v in enumerate(smooth)
        ]

        if len(pts) > 1:
            pygame.draw.lines(
                screen,
                GOLD,
                False,
                pts,
                2,
            )

        pygame.draw.rect(
            screen,
            (35, 42, 65),
            (
                PAD,
                PAD,
                cw,
                ch,
            ),
            1,
        )

        title = font_body.render(
            "Skill Improvement over Training",
            True,
            TEXT_HI,
        )

        screen.blit(
            title,
            (
                SCREEN_W // 2
                - title.get_width() // 2,
                14,
            ),
        )

        stats = (
            f"Episodes: {n}   "
            f"Max: {max(vals)}   "
            f"Rolling avg ({window}): "
            f"{np.mean(vals[-window:]):.1f}"
        )

        screen.blit(
            font_sm.render(
                stats,
                True,
                TEXT_MID,
            ),
            (
                PAD,
                SCREEN_H - 28,
            ),
        )

        hint = font_sm.render(
            "ESC  back to menu",
            True,
            TEXT_LO,
        )

        screen.blit(
            hint,
            (
                SCREEN_W
                - hint.get_width()
                - 16,
                SCREEN_H - 28,
            ),
        )

        pygame.display.flip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    pygame.init()

    screen = pygame.display.set_mode(
        (SCREEN_W, SCREEN_H)
    )

    pygame.display.set_caption(
        "Precision Platformer Gauntlet"
    )

    fonts = (
        pygame.font.SysFont(
            "Arial",
            28,
            bold=True,
        ),
        pygame.font.SysFont(
            "Arial",
            20,
            bold=True,
        ),
        pygame.font.SysFont(
            "Arial",
            14,
        ),
    )

    while True:

        mode = run_menu(
            screen,
            fonts,
        )

        if mode == "ai":
            run_ai_mode(
                screen,
                fonts,
            )

        elif mode == "human":
            run_human_mode(
                screen,
                fonts,
            )

        elif mode == "chart":
            run_chart_mode(
                screen,
                fonts,
            )


if __name__ == "__main__":
    main()