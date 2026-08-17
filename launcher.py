"""
launcher.py  –  Precision Platformer Gauntlet
----------------------------------------------
Modes:
  WATCH AI   – trained agent plays automatically
  PLAY       – human keyboard control
  TRAINING   – in-window improvement chart

Controls (human mode):
  LEFT / RIGHT   move
  SPACE or UP    jump
  ESC            back to menu
"""

import os, sys, math, time
import pygame
import numpy as np

from platformer_env import (
    PlatformerGauntletEnv,
    SCREEN_W, SCREEN_H,
    AGENT_W, AGENT_H,
    GRAVITY, JUMP_VELOCITY, MOVE_SPEED, TERMINAL_VY,
    PLATFORM_THICKNESS, generate_level,
)

# ── palette ───────────────────────────────────────────────────────────────────
BG         = (10,  12,  20)
PANEL      = (18,  22,  36)
BORDER     = (35,  42,  65)
ACCENT     = (80, 200, 130)   # cleared platforms / highlights
PLATFORM_C = (60,  85, 160)   # upcoming platforms
AGENT_COL  = (220, 190,  55)  # agent body
TEXT_HI    = (220, 220, 225)
TEXT_MID   = (140, 145, 165)
TEXT_LO    = ( 80,  88, 110)
DANGER     = (200,  65,  65)
GOLD       = (220, 175,  45)

FPS = 60


# ── model loading ─────────────────────────────────────────────────────────────
def load_model():
    try:
        from stable_baselines3 import PPO
        if os.path.exists("platformer_model.zip"):
            return PPO.load("platformer_model.zip")
        ckpt = "checkpoints"
        if os.path.isdir(ckpt):
            zips = sorted(
                [f for f in os.listdir(ckpt) if f.endswith(".zip")],
                key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
            if zips:
                return PPO.load(os.path.join(ckpt, zips[-1]))
    except Exception:
        pass
    return None


# ── collision (AABB, not centre-point) ────────────────────────────────────────
def check_landing(ax, ay, prev_ay, vy, platforms):
    """
    Returns (landed, new_ay, new_vy, platform_idx) or (False, ay, vy, -1).
    Uses full AABB overlap on x-axis instead of a single centre point,
    which prevents phasing through platform edges.
    """
    if vy < 0:
        return False, ay, vy, -1

    feet_y      = ay      + AGENT_H
    prev_feet_y = prev_ay + AGENT_H
    agent_left  = ax
    agent_right = ax + AGENT_W

    for idx, p in enumerate(platforms):
        # AABB x-overlap: any part of agent over platform
        if agent_right > p.x and agent_left < p.x_end:
            if feet_y >= p.y and prev_feet_y <= p.y:
                return True, p.y - AGENT_H, 0.0, idx

    return False, ay, vy, -1


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_bg(surface):
    surface.fill(BG)


def draw_level(surface, platforms, cam_x, furthest_idx):
    for idx, p in enumerate(platforms):
        sx = p.x - cam_x
        if sx + p.width < 0 or sx > SCREEN_W:
            continue
        col = ACCENT if idx <= furthest_idx else PLATFORM_C
        pygame.draw.rect(surface, col,
                         (int(sx), int(p.y), int(p.width), PLATFORM_THICKNESS))


def draw_agent(surface, ax, ay):
    # clean rectangle, no decorations
    pygame.draw.rect(surface, AGENT_COL,
                     (int(ax), int(ay), AGENT_W, AGENT_H))


def draw_progress_bar(surface, font, furthest, total):
    bar_w, bar_h = 240, 8
    bx = SCREEN_W - bar_w - 16
    by = 16
    pct = furthest / max(total, 1)
    pygame.draw.rect(surface, PANEL,  (bx, by, bar_w, bar_h), border_radius=4)
    if pct > 0:
        pygame.draw.rect(surface, ACCENT,
                         (bx, by, int(bar_w * pct), bar_h), border_radius=4)
    pygame.draw.rect(surface, BORDER, (bx, by, bar_w, bar_h), 1, border_radius=4)
    lbl = font.render(f"{furthest} / {total}", True, TEXT_MID)
    surface.blit(lbl, (bx + bar_w - lbl.get_width(), by + bar_h + 4))


def draw_hint(surface, font, text):
    lbl = font.render(text, True, TEXT_LO)
    surface.blit(lbl, (16, 16))


# ── MENU ──────────────────────────────────────────────────────────────────────
def run_menu(screen, fonts, has_model, has_log):
    font_title, font_body, font_sm = fonts
    clock   = pygame.time.Clock()

    options = [
        ("Watch AI Play",  has_model, "no model found — run train.py first", "ai"),
        ("Play Yourself",  True,      "",                                     "human"),
        ("Training Chart", has_log,   "no training_log.csv found",            "chart"),
    ]
    sel = 0

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP,   pygame.K_w):
                    sel = (sel - 1) % len(options)
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(options)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    name, enabled, _, mode = options[sel]
                    if enabled:
                        return mode
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

        draw_bg(screen)

        # title block
        title = font_title.render("PRECISION PLATFORMER GAUNTLET", True, TEXT_HI)
        sub   = font_sm.render("Neural Network Learning Demo", True, TEXT_MID)
        ty    = 72
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, ty))
        screen.blit(sub,   (SCREEN_W // 2 - sub.get_width()   // 2,
                             ty + title.get_height() + 6))

        # thin divider
        div_y = ty + title.get_height() + sub.get_height() + 24
        pygame.draw.line(screen, BORDER,
                         (SCREEN_W // 4, div_y), (SCREEN_W * 3 // 4, div_y))

        # menu items
        item_h  = 64
        start_y = div_y + 40
        iw      = 400

        for i, (name, enabled, warn, _) in enumerate(options):
            iy     = start_y + i * (item_h + 12)
            ix     = SCREEN_W // 2 - iw // 2
            is_sel = i == sel

            bg_col = PANEL if is_sel else BG
            pygame.draw.rect(screen, bg_col,
                             (ix, iy, iw, item_h), border_radius=6)
            pygame.draw.rect(screen, ACCENT if is_sel else BORDER,
                             (ix, iy, iw, item_h), 1, border_radius=6)

            # left accent bar on selected
            if is_sel:
                pygame.draw.rect(screen, ACCENT, (ix, iy + 10, 3, item_h - 20))

            name_col = TEXT_HI  if enabled else TEXT_LO
            name_surf = font_body.render(name, True, name_col)
            screen.blit(name_surf, (ix + 20, iy + item_h // 2 - name_surf.get_height() // 2))

            if not enabled and warn:
                w_surf = font_sm.render(warn, True, DANGER)
                screen.blit(w_surf, (ix + iw - w_surf.get_width() - 16,
                                     iy + item_h // 2 - w_surf.get_height() // 2))

        # footer
        foot = font_sm.render(
            "W / S  or  ↑ ↓   navigate      ENTER  select      ESC  quit",
            True, TEXT_LO)
        screen.blit(foot, (SCREEN_W // 2 - foot.get_width() // 2, SCREEN_H - 32))

        pygame.display.flip()


# ── AI WATCH ──────────────────────────────────────────────────────────────────
def run_ai_mode(screen, fonts, model):
    _, font_body, font_sm = fonts
    clock   = pygame.time.Clock()
    env     = PlatformerGauntletEnv(render_mode=None, seed=0)
    episode = 0

    while True:
        episode += 1
        obs, _ = env.reset(seed=episode)
        done   = False

        while not done:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    env.close(); return

            action, _ = model.predict(obs, deterministic=True) if model \
                        else (env.action_space.sample(), None)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            cam_x = env.agent_x - SCREEN_W * 0.3
            draw_bg(screen)
            draw_level(screen, env.platforms, cam_x, env.furthest_platform_idx)
            draw_agent(screen, env.agent_x - cam_x, env.agent_y)
            draw_progress_bar(screen, font_sm,
                              env.furthest_platform_idx, len(env.platforms) - 1)
            draw_hint(screen, font_sm,
                      f"Episode {episode}   ESC → menu")
            pygame.display.flip()

        time.sleep(0.5)


# ── HUMAN PLAY ────────────────────────────────────────────────────────────────
def run_human_mode(screen, fonts):
    _, font_body, font_sm = fonts
    clock        = pygame.time.Clock()
    best_ever    = 0
    episode      = 0

    while True:
        episode  += 1
        platforms = generate_level(seed=0)
        start     = platforms[0]
        ax        = start.x + start.width / 2 - AGENT_W / 2
        ay        = start.y - AGENT_H
        vx = vy   = 0.0
        prev_ay   = ay
        on_ground = True
        furthest  = 0
        t         = 0

        running = True
        while running:
            clock.tick(FPS)
            t += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            keys = pygame.key.get_pressed()
            vx   = (-MOVE_SPEED if keys[pygame.K_LEFT]  else
                     MOVE_SPEED if keys[pygame.K_RIGHT] else 0.0)

            if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) and on_ground:
                vy        = JUMP_VELOCITY
                on_ground = False

            vy      = min(vy + GRAVITY, TERMINAL_VY)
            prev_ay = ay
            ax     += vx
            ay     += vy

            for p in platforms:
                p.update(t)

            landed, ay, vy, plat_idx = check_landing(ax, ay, prev_ay, vy, platforms)
            if landed:
                on_ground = True
                if plat_idx > furthest:
                    furthest  = plat_idx
                    best_ever = max(best_ever, furthest)
            else:
                on_ground = False

            cam_x = ax - SCREEN_W * 0.3
            draw_bg(screen)
            draw_level(screen, platforms, cam_x, furthest)
            draw_agent(screen, ax - cam_x, ay)
            draw_progress_bar(screen, font_sm, furthest, len(platforms) - 1)
            draw_hint(screen, font_sm,
                      f"← → move   SPACE jump   ESC menu   Best: {best_ever}")
            pygame.display.flip()

            if ay > SCREEN_H + 100:
                running = False

        # death overlay
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        lines = [
            (font_body, f"Reached platform {furthest} / {len(platforms)-1}", TEXT_HI),
            (font_sm,   f"Personal best:  {best_ever}",                      TEXT_MID),
            (font_sm,   "ENTER  retry       ESC  menu",                       TEXT_LO),
        ]
        total_h = sum(f.render(t, True, (0,0,0)).get_height() + 10 for f, t, _ in lines)
        y = SCREEN_H // 2 - total_h // 2
        for fnt, txt, col in lines:
            surf = fnt.render(txt, True, col)
            screen.blit(surf, (SCREEN_W // 2 - surf.get_width() // 2, y))
            y += surf.get_height() + 10

        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN: waiting = False
                    if event.key == pygame.K_ESCAPE: return


# ── CHART ─────────────────────────────────────────────────────────────────────
def run_chart_mode(screen, fonts):
    _, font_body, font_sm = fonts
    clock = pygame.time.Clock()

    import csv
    try:
        with open("training_log.csv") as f:
            rows  = list(csv.DictReader(f))
            vals  = [int(r["furthest_platform"]) for r in rows]
            total = int(rows[0]["total_platforms"]) if rows else 40
    except Exception:
        vals, total = [], 40

    if not vals:
        while True:
            draw_bg(screen)
            msg = font_body.render("No training_log.csv — run train.py first",
                                   True, DANGER)
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            clock.tick(FPS)

    window = max(1, len(vals) // 20)
    kernel = np.ones(window) / window
    smooth = np.convolve(vals, kernel, mode="valid")

    PAD = 64
    cw  = SCREEN_W - PAD * 2
    ch  = SCREEN_H - PAD * 2 - 48

    def to_px(i, v, n):
        x = PAD + int(i / max(n - 1, 1) * cw)
        y = PAD + int((1 - v / max(total, 1)) * ch)
        return x, y

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return

        draw_bg(screen)

        # y grid lines
        for tick in range(0, total + 1, 5):
            gy = PAD + int((1 - tick / total) * ch)
            pygame.draw.line(screen, PANEL, (PAD, gy), (PAD + cw, gy))
            lbl = font_sm.render(str(tick), True, TEXT_LO)
            screen.blit(lbl, (PAD - lbl.get_width() - 6, gy - 8))

        # full-level reference line
        pygame.draw.line(screen, (*ACCENT, 120), (PAD, PAD), (PAD + cw, PAD), 1)

        # scatter
        n = len(vals)
        for i, v in enumerate(vals):
            px, py = to_px(i, v, n)
            pygame.draw.circle(screen, (*PLATFORM_C, 100), (px, py), 2)

        # rolling average
        off = (n - len(smooth)) // 2
        pts = [to_px(off + i, v, n) for i, v in enumerate(smooth)]
        if len(pts) > 1:
            pygame.draw.lines(screen, GOLD, False, pts, 2)

        # axes border
        pygame.draw.rect(screen, BORDER, (PAD, PAD, cw, ch), 1)

        # labels
        title = font_body.render("Skill Improvement over Training", True, TEXT_HI)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 14))

        stats = (f"Episodes: {n}   Max platform: {max(vals)}"
                 f"   Rolling avg ({window} eps): {np.mean(vals[-window:]):.1f}")
        st = font_sm.render(stats, True, TEXT_MID)
        screen.blit(st, (PAD, SCREEN_H - 28))

        legend_items = [
            (GOLD,       "rolling average"),
            (PLATFORM_C, "per episode"),
            (ACCENT,     "full level"),
        ]
        lx = SCREEN_W - 200
        for i, (col, lbl) in enumerate(legend_items):
            pygame.draw.rect(screen, col, (lx, 16 + i * 18, 14, 3))
            ls = font_sm.render(lbl, True, TEXT_MID)
            screen.blit(ls, (lx + 20, 10 + i * 18))

        hint = font_sm.render("ESC  back to menu", True, TEXT_LO)
        screen.blit(hint, (SCREEN_W - hint.get_width() - 16, SCREEN_H - 28))

        pygame.display.flip()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Precision Platformer Gauntlet")

    font_title = pygame.font.SysFont("Arial",  28, bold=True)
    font_body  = pygame.font.SysFont("Arial",  20, bold=True)
    font_sm    = pygame.font.SysFont("Arial",  14)
    fonts      = (font_title, font_body, font_sm)

    model     = load_model()
    has_model = model is not None
    has_log   = os.path.exists("training_log.csv")

    while True:
        mode = run_menu(screen, fonts, has_model, has_log)
        if mode == "ai":
            run_ai_mode(screen, fonts, model)
        elif mode == "human":
            run_human_mode(screen, fonts)
        elif mode == "chart":
            run_chart_mode(screen, fonts)


if __name__ == "__main__":
    main()    main()