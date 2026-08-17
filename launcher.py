"""
launcher.py  –  Precision Platformer Gauntlet
----------------------------------------------
A full-screen pygame launcher with three modes:
  1. WATCH AI   – watch the trained agent play
  2. PLAY       – human keyboard control
  3. TRAINING   – view the improvement chart

Controls (human mode):
  Arrow LEFT / RIGHT  – move
  SPACE               – jump
  ESC                 – back to menu
"""

import os
import sys
import math
import time

import pygame
import numpy as np

from platformer_env import (
    PlatformerGauntletEnv,
    SCREEN_W, SCREEN_H,
    AGENT_W, AGENT_H,
    GRAVITY, JUMP_VELOCITY, MOVE_SPEED, TERMINAL_VY,
    PLATFORM_THICKNESS, generate_level,
)

# ── palette ──────────────────────────────────────────────────────────────────
BG        = (12,  14,  23)   # near-black blue
PANEL     = (20,  24,  40)   # card background
ACCENT    = (99, 210, 140)   # mint green  (platforms cleared)
PLATFORM  = (75,  95, 180)   # slate blue  (platforms ahead)
AGENT_COL = (240,200,  60)   # warm yellow
TEXT_HI   = (230,230,230)
TEXT_LO   = (110,120,150)
DANGER    = (220, 80,  80)
GOLD      = (255,200,  60)

FPS = 60


# ── helpers ───────────────────────────────────────────────────────────────────
def load_model():
    """Try to import SB3 and load the trained model; return None if missing."""
    try:
        from stable_baselines3 import PPO
        if os.path.exists("platformer_model.zip"):
            return PPO.load("platformer_model.zip")
        # try latest checkpoint
        ckpt_dir = "checkpoints"
        if os.path.isdir(ckpt_dir):
            zips = sorted(
                [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")],
                key=lambda f: int(''.join(filter(str.isdigit, f)) or 0)
            )
            if zips:
                return PPO.load(os.path.join(ckpt_dir, zips[-1]))
    except Exception:
        pass
    return None


def draw_level(surface, platforms, cam_x, furthest_idx):
    for idx, p in enumerate(platforms):
        sx = p.x - cam_x
        if sx + p.width < -20 or sx > SCREEN_W + 20:
            continue
        color = ACCENT if idx <= furthest_idx else PLATFORM
        # platform body
        pygame.draw.rect(surface, color,
                         (sx, p.y, p.width, PLATFORM_THICKNESS))
        # subtle top highlight
        pygame.draw.rect(surface, tuple(min(255, c+40) for c in color),
                         (sx, p.y, p.width, 2))


def draw_agent(surface, ax, ay, on_ground):
    # body
    pygame.draw.rect(surface, AGENT_COL, (ax, ay, AGENT_W, AGENT_H), border_radius=4)
    # eyes
    eye_y = ay + 8
    pygame.draw.circle(surface, BG, (int(ax + 8),  int(eye_y)), 4)
    pygame.draw.circle(surface, BG, (int(ax + 18), int(eye_y)), 4)
    pygame.draw.circle(surface, TEXT_HI, (int(ax + 9),  int(eye_y)), 2)
    pygame.draw.circle(surface, TEXT_HI, (int(ax + 19), int(eye_y)), 2)
    # shadow when in air
    if not on_ground:
        shadow = pygame.Surface((AGENT_W, 6), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 60))
        surface.blit(shadow, (ax, ay + AGENT_H + 2))


def draw_hud(surface, font_big, font_sm, furthest, total, step, extra=""):
    # platform counter
    pct  = furthest / max(total, 1)
    bar_w = 260
    bar_h = 10
    bx, by = SCREEN_W - bar_w - 16, 14
    pygame.draw.rect(surface, PANEL,  (bx, by, bar_w, bar_h), border_radius=5)
    pygame.draw.rect(surface, ACCENT, (bx, by, int(bar_w * pct), bar_h), border_radius=5)
    label = font_sm.render(f"Platform  {furthest} / {total}", True, TEXT_HI)
    surface.blit(label, (bx, by + 14))
    if extra:
        note = font_sm.render(extra, True, TEXT_LO)
        surface.blit(note, (16, 14))


def parallax_bg(surface, t):
    """Subtle scrolling star field."""
    rng = np.random.default_rng(42)
    stars = rng.integers(0, [SCREEN_W * 3, SCREEN_H], size=(120, 2))
    speeds = rng.uniform(0.1, 0.5, 120)
    for (sx, sy), sp in zip(stars, speeds):
        x = int(sx - t * sp * 0.3) % SCREEN_W
        brightness = rng.integers(80, 180)
        pygame.draw.circle(surface, (brightness,)*3, (x, int(sy)), 1)


# ── MENU ──────────────────────────────────────────────────────────────────────
def run_menu(screen, fonts):
    font_title, font_big, font_sm = fonts
    clock  = pygame.time.Clock()
    t      = 0
    has_model = os.path.exists("platformer_model.zip") or os.path.isdir("checkpoints")
    has_log   = os.path.exists("training_log.csv")

    options = [
        ("WATCH AI PLAY",    "See the trained agent tackle the gauntlet",  has_model, "ai"),
        ("PLAY YOURSELF",    "Take control — how far can you get?",         True,      "human"),
        ("TRAINING CHART",   "View the agent's learning curve",             has_log,   "chart"),
    ]

    selected = 0
    while True:
        dt = clock.tick(FPS)
        t += dt * 0.001

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    selected = (selected - 1) % len(options)
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    selected = (selected + 1) % len(options)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    _, _, enabled, mode = options[selected]
                    if enabled:
                        return mode
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

        screen.fill(BG)
        parallax_bg(screen, t * 60)

        # title
        title = font_title.render("PRECISION PLATFORMER", True, TEXT_HI)
        sub   = font_sm.render("Neural Network Gauntlet", True, ACCENT)
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 80))
        screen.blit(sub,   (SCREEN_W//2 - sub.get_width()//2,   80 + title.get_height() + 8))

        # animated divider
        div_y = 180
        wave_w = int(abs(math.sin(t * 1.5)) * 40 + 200)
        pygame.draw.rect(screen, ACCENT,
                         (SCREEN_W//2 - wave_w//2, div_y, wave_w, 2))

        # menu cards
        card_w, card_h = 520, 88
        card_x = SCREEN_W//2 - card_w//2
        start_y = 220

        for i, (name, desc, enabled, _) in enumerate(options):
            cy    = start_y + i * (card_h + 16)
            is_sel = i == selected
            alpha  = 255 if enabled else 100

            # card bg
            card_col = PANEL if not is_sel else (30, 38, 65)
            pygame.draw.rect(screen, card_col,
                             (card_x, cy, card_w, card_h), border_radius=10)
            if is_sel:
                pygame.draw.rect(screen, ACCENT,
                                 (card_x, cy, card_w, card_h), 2, border_radius=10)
            if not enabled:
                pygame.draw.rect(screen, (40, 40, 50),
                                 (card_x, cy, card_w, card_h), border_radius=10)

            # text
            col_name = TEXT_HI if enabled else TEXT_LO
            col_desc = ACCENT  if (is_sel and enabled) else TEXT_LO
            name_surf = font_big.render(name, True, col_name)
            desc_surf = font_sm.render(desc if enabled else desc + "  [no model found]",
                                       True, col_desc)
            screen.blit(name_surf, (card_x + 24, cy + 14))
            screen.blit(desc_surf, (card_x + 24, cy + 14 + name_surf.get_height() + 4))

            # arrow
            if is_sel and enabled:
                ax = card_x + card_w - 36
                ay = cy + card_h//2
                pts = [(ax, ay-8), (ax+14, ay), (ax, ay+8)]
                pygame.draw.polygon(screen, ACCENT, pts)

        # footer
        foot = font_sm.render("↑ ↓  navigate      ENTER  select      ESC  quit",
                               True, TEXT_LO)
        screen.blit(foot, (SCREEN_W//2 - foot.get_width()//2, SCREEN_H - 36))

        pygame.display.flip()


# ── AI PLAYBACK ───────────────────────────────────────────────────────────────
def run_ai_mode(screen, fonts, model):
    _, font_big, font_sm = fonts
    clock = pygame.time.Clock()
    env   = PlatformerGauntletEnv(render_mode=None, seed=0)

    episode = 0
    while True:
        episode += 1
        obs, info = env.reset(seed=episode)
        done = False
        t    = 0

        while not done:
            t += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close(); return

            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            cam_x = env.agent_x - SCREEN_W * 0.3
            screen.fill(BG)
            parallax_bg(screen, t)
            draw_level(screen, env.platforms, cam_x, env.furthest_platform_idx)
            ax = env.agent_x - cam_x
            draw_agent(screen, ax, env.agent_y, env.on_ground)
            draw_hud(screen, font_big, font_sm,
                     env.furthest_platform_idx,
                     len(env.platforms) - 1,
                     env.steps,
                     extra=f"Episode {episode}   ESC = menu")
            pygame.display.flip()
            clock.tick(FPS)

        # brief pause between episodes
        time.sleep(0.6)


# ── HUMAN PLAY ────────────────────────────────────────────────────────────────
def run_human_mode(screen, fonts):
    _, font_big, font_sm = fonts
    clock = pygame.time.Clock()

    best_platform = 0
    episode       = 0

    while True:
        episode += 1
        platforms = generate_level(seed=0)
        start     = platforms[0]
        ax = start.x + start.width / 2 - AGENT_W / 2
        ay = start.y - AGENT_H
        vx = vy = 0.0
        on_ground = True
        furthest  = 0
        cur_plat  = 0
        furthest_x = ax
        t = steps = 0
        done = False
        death_msg = ""

        while not done:
            steps += 1
            t     += 1
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                vx = -MOVE_SPEED
            elif keys[pygame.K_RIGHT]:
                vx = MOVE_SPEED
            else:
                vx = 0.0

            if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) and on_ground:
                vy        = JUMP_VELOCITY
                on_ground = False

            vy   = min(vy + GRAVITY, TERMINAL_VY)
            prev_ay = ay
            ax  += vx
            ay  += vy

            # update moving platforms
            for p in platforms:
                p.update(t)

            # collision
            landed = False
            if vy >= 0:
                feet_y      = ay + AGENT_H
                prev_feet_y = prev_ay + AGENT_H
                mid_x       = ax + AGENT_W * 0.5
                for idx, p in enumerate(platforms):
                    if p.x <= mid_x <= p.x_end:
                        if feet_y >= p.y and prev_feet_y <= p.y:
                            ay        = p.y - AGENT_H
                            vy        = 0.0
                            on_ground = True
                            cur_plat  = idx
                            landed    = True
                            if idx > furthest:
                                furthest = idx
                                best_platform = max(best_platform, idx)
                            break
            if not landed:
                on_ground = False

            # void death
            if ay > SCREEN_H + 100:
                done      = True
                death_msg = f"Reached platform {furthest}  —  best ever: {best_platform}"

            cam_x = ax - SCREEN_W * 0.3
            screen.fill(BG)
            parallax_bg(screen, t)
            draw_level(screen, platforms, cam_x, furthest)
            draw_agent(screen, ax - cam_x, ay, on_ground)
            draw_hud(screen, font_big, font_sm,
                     furthest, len(platforms) - 1, steps,
                     extra=f"SPACE/↑ jump   ←→ move   ESC menu   Best: {best_platform}")
            pygame.display.flip()

        # death screen
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))
        msg1 = font_big.render("YOU FELL", True, DANGER)
        msg2 = font_sm.render(death_msg, True, TEXT_HI)
        msg3 = font_sm.render("ENTER to retry   ESC for menu", True, TEXT_LO)
        screen.blit(msg1, (SCREEN_W//2 - msg1.get_width()//2, SCREEN_H//2 - 60))
        screen.blit(msg2, (SCREEN_W//2 - msg2.get_width()//2, SCREEN_H//2))
        screen.blit(msg3, (SCREEN_W//2 - msg3.get_width()//2, SCREEN_H//2 + 50))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting = False
                    if event.key == pygame.K_ESCAPE:
                        return


# ── CHART ─────────────────────────────────────────────────────────────────────
def run_chart_mode(screen, fonts):
    """Render a live training chart directly in pygame (no matplotlib window)."""
    _, font_big, font_sm = fonts
    clock = pygame.time.Clock()

    import csv
    try:
        with open("training_log.csv") as f:
            rows   = list(csv.DictReader(f))
            vals   = [int(r["furthest_platform"]) for r in rows]
            total  = int(rows[0]["total_platforms"]) if rows else 40
    except Exception:
        vals, total = [], 40

    if not vals:
        waiting = True
        while waiting:
            screen.fill(BG)
            msg = font_big.render("No training_log.csv found — run train.py first",
                                  True, DANGER)
            screen.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type in (pygame.QUIT,) or (
                   event.type == pygame.KEYDOWN and
                   event.key == pygame.K_ESCAPE):
                    return
            clock.tick(FPS)
        return

    # rolling average
    window = max(1, len(vals) // 20)
    kernel = np.ones(window) / window
    smooth = np.convolve(vals, kernel, mode="valid")

    PAD = 60
    cw  = SCREEN_W - PAD * 2
    ch  = SCREEN_H - PAD * 2 - 60

    def to_px(i, v):
        x = PAD + int(i / max(len(vals)-1, 1) * cw)
        y = PAD + 40 + int((1 - v / total) * ch)
        return x, y

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(BG)

        # grid lines
        for tick in range(0, total + 1, 10):
            y = PAD + 40 + int((1 - tick / total) * ch)
            pygame.draw.line(screen, PANEL, (PAD, y), (PAD + cw, y))
            lbl = font_sm.render(str(tick), True, TEXT_LO)
            screen.blit(lbl, (PAD - lbl.get_width() - 6, y - 8))

        # scatter dots
        for i, v in enumerate(vals):
            px, py = to_px(i, v)
            pygame.draw.circle(screen, (*PLATFORM, 120), (px, py), 2)

        # rolling average line
        offset = (len(vals) - len(smooth)) // 2
        pts = [to_px(offset + i, v) for i, v in enumerate(smooth)]
        if len(pts) > 1:
            pygame.draw.lines(screen, GOLD, False, pts, 2)

        # full level line
        pygame.draw.line(screen, ACCENT,
                         (PAD, PAD + 40), (PAD + cw, PAD + 40), 1)
        lbl = font_sm.render(f"full level ({total})", True, ACCENT)
        screen.blit(lbl, (PAD + cw - lbl.get_width(), PAD + 44))

        # labels
        title = font_big.render("Skill Improvement over Training", True, TEXT_HI)
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 14))

        xl = font_sm.render("Episode →", True, TEXT_LO)
        screen.blit(xl, (SCREEN_W//2 - xl.get_width()//2, SCREEN_H - 28))

        yl = font_sm.render("Furthest Platform", True, TEXT_LO)
        yl = pygame.transform.rotate(yl, 90)
        screen.blit(yl, (10, SCREEN_H//2 - yl.get_height()//2))

        stats = (f"Episodes: {len(vals)}   "
                 f"Max: {max(vals)}   "
                 f"Avg (last {window}): {np.mean(vals[-window:]):.1f}")
        st = font_sm.render(stats, True, ACCENT)
        screen.blit(st, (PAD, SCREEN_H - 28))

        esc_hint = font_sm.render("ESC  back to menu", True, TEXT_LO)
        screen.blit(esc_hint, (SCREEN_W - esc_hint.get_width() - 16, SCREEN_H - 28))

        pygame.display.flip()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Precision Platformer Gauntlet")

    # fonts
    font_title = pygame.font.SysFont("Arial Black", 42, bold=True)
    font_big   = pygame.font.SysFont("Arial",       22, bold=True)
    font_sm    = pygame.font.SysFont("Arial",        16)
    fonts      = (font_title, font_big, font_sm)

    model = load_model()

    while True:
        mode = run_menu(screen, fonts)
        if mode == "ai":
            run_ai_mode(screen, fonts, model)
        elif mode == "human":
            run_human_mode(screen, fonts)
        elif mode == "chart":
            run_chart_mode(screen, fonts)


if __name__ == "__main__":
    main()