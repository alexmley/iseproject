"""
launcher.py  –  Precision Platformer Gauntlet - v3, with speed options now
----------------------------------------------
Modes:
  WATCH AI   – trained agent plays (speed control + checkpoint picker)
  PLAY       – human keyboard control
  TRAINING   – in-window improvement chart

Controls (human mode):
  LEFT / RIGHT     move
  SPACE or UP      jump
  ESC              back to menu

Controls (AI watch mode):
  1 / 2 / 3 / 4   set speed (1x / 2x / 5x / max)
  ESC              back to menu
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
ACCENT     = (80, 200, 130)
PLATFORM_C = (60,  85, 160)
AGENT_COL  = (220, 190,  55)
TEXT_HI    = (220, 220, 225)
TEXT_MID   = (140, 145, 165)
TEXT_LO    = ( 80,  88, 110)
DANGER     = (200,  65,  65)
GOLD       = (220, 175,  45)

FPS = 60

# Speed multipliers for AI watch mode
SPEEDS     = [1, 2, 5, 0]        # 0 = uncapped (max)
SPEED_LBLS = ["1×", "2×", "5×", "MAX"]


# ── model / checkpoint helpers ────────────────────────────────────────────────
def list_checkpoints():
    """Return sorted list of (label, path) for all available models."""
    options = []
    ckpt = "checkpoints"
    if os.path.isdir(ckpt):
        zips = sorted(
            [f for f in os.listdir(ckpt) if f.endswith(".zip")],
            key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
        for z in zips:
            steps = int("".join(filter(str.isdigit, z)) or 0)
            label = f"{steps:,} steps"
            options.append((label, os.path.join(ckpt, z)))
    if os.path.exists("platformer_model.zip"):
        options.append(("Final model", "platformer_model.zip"))
    return options


def load_model_from_path(path):
    try:
        from stable_baselines3 import PPO
        return PPO.load(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None


# ── collision (full AABB) ─────────────────────────────────────────────────────
def check_landing(ax, ay, prev_ay, vy, platforms):
    if vy < 0:
        return False, ay, vy, -1
    feet_y      = ay      + AGENT_H
    prev_feet_y = prev_ay + AGENT_H
    agent_left  = ax
    agent_right = ax + AGENT_W
    for idx, p in enumerate(platforms):
        if agent_right > p.x and agent_left < p.x_end:
            if feet_y >= p.y and prev_feet_y <= p.y:
                return True, p.y - AGENT_H, 0.0, idx
    return False, ay, vy, -1


# ── shared draw helpers ───────────────────────────────────────────────────────
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
    lbl = font.render(f"Platform  {furthest} / {total}", True, TEXT_MID)
    surface.blit(lbl, (bx + bar_w - lbl.get_width(), by + bar_h + 6))


def draw_hint(surface, font, text, y=16):
    lbl = font.render(text, True, TEXT_LO)
    surface.blit(lbl, (16, y))


def draw_speed_indicator(surface, font, speed_idx):
    """Row of speed buttons in top-centre."""
    labels = SPEED_LBLS
    btn_w, btn_h, gap = 52, 22, 6
    total_w = len(labels) * btn_w + (len(labels) - 1) * gap
    sx = SCREEN_W // 2 - total_w // 2
    sy = 14
    for i, lbl in enumerate(labels):
        bx = sx + i * (btn_w + gap)
        active = i == speed_idx
        col_bg  = ACCENT  if active else PANEL
        col_txt = BG      if active else TEXT_LO
        pygame.draw.rect(surface, col_bg,  (bx, sy, btn_w, btn_h), border_radius=4)
        pygame.draw.rect(surface, BORDER,  (bx, sy, btn_w, btn_h), 1, border_radius=4)
        t = font.render(lbl, True, col_txt)
        surface.blit(t, (bx + btn_w // 2 - t.get_width() // 2,
                         sy + btn_h // 2 - t.get_height() // 2))
    hint = font.render("1 2 3 4  speed", True, TEXT_LO)
    surface.blit(hint, (sx + total_w + 10, sy + 4))


# ── MENU ──────────────────────────────────────────────────────────────────────
def run_menu(screen, fonts, has_model, has_log):
    font_title, font_body, font_sm = fonts
    clock = pygame.time.Clock()

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
                    _, enabled, _, mode = options[sel]
                    if enabled:
                        return mode
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

        draw_bg(screen)

        title = font_title.render("PRECISION PLATFORMER GAUNTLET", True, TEXT_HI)
        sub   = font_sm.render("Neural Network Learning Demo", True, TEXT_MID)
        ty    = 72
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, ty))
        screen.blit(sub,   (SCREEN_W // 2 - sub.get_width()   // 2,
                             ty + title.get_height() + 6))

        div_y = ty + title.get_height() + sub.get_height() + 24
        pygame.draw.line(screen, BORDER,
                         (SCREEN_W // 4, div_y), (SCREEN_W * 3 // 4, div_y))

        item_h, iw = 64, 400
        start_y    = div_y + 40

        for i, (name, enabled, warn, _) in enumerate(options):
            iy     = start_y + i * (item_h + 12)
            ix     = SCREEN_W // 2 - iw // 2
            is_sel = i == sel

            pygame.draw.rect(screen, PANEL if is_sel else BG,
                             (ix, iy, iw, item_h), border_radius=6)
            pygame.draw.rect(screen, ACCENT if is_sel else BORDER,
                             (ix, iy, iw, item_h), 1, border_radius=6)
            if is_sel:
                pygame.draw.rect(screen, ACCENT, (ix, iy + 10, 3, item_h - 20))

            ns = font_body.render(name, True, TEXT_HI if enabled else TEXT_LO)
            screen.blit(ns, (ix + 20, iy + item_h // 2 - ns.get_height() // 2))

            if not enabled and warn:
                ws = font_sm.render(warn, True, DANGER)
                screen.blit(ws, (ix + iw - ws.get_width() - 16,
                                 iy + item_h // 2 - ws.get_height() // 2))

        foot = font_sm.render(
            "↑ ↓  navigate      ENTER  select      ESC  quit", True, TEXT_LO)
        screen.blit(foot, (SCREEN_W // 2 - foot.get_width() // 2, SCREEN_H - 32))
        pygame.display.flip()


# ── CHECKPOINT PICKER ─────────────────────────────────────────────────────────
def run_checkpoint_picker(screen, fonts):
    """
    Full-screen selector showing all available checkpoints + final model.
    Returns (label, path) of chosen checkpoint, or None if ESC pressed.
    """
    font_title, font_body, font_sm = fonts
    clock      = pygame.time.Clock()
    checkpoints = list_checkpoints()

    if not checkpoints:
        return None

    sel    = len(checkpoints) - 1   # default to latest
    scroll = 0
    VISIBLE = 8
    item_h  = 52

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    sel    = max(0, sel - 1)
                    scroll = min(scroll, sel)
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    sel    = min(len(checkpoints) - 1, sel + 1)
                    scroll = max(scroll, sel - VISIBLE + 1)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return checkpoints[sel]
                if event.key == pygame.K_ESCAPE:
                    return None

        draw_bg(screen)

        title = font_title.render("SELECT CHECKPOINT", True, TEXT_HI)
        sub   = font_sm.render("Choose which training snapshot to watch", True, TEXT_MID)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 60))
        screen.blit(sub,   (SCREEN_W // 2 - sub.get_width()   // 2, 60 + title.get_height() + 6))

        iw   = 500
        ix   = SCREEN_W // 2 - iw // 2
        sy   = 140

        visible_items = checkpoints[scroll: scroll + VISIBLE]
        for i, (label, path) in enumerate(visible_items):
            real_idx = scroll + i
            iy       = sy + i * (item_h + 8)
            is_sel   = real_idx == sel
            is_final = path == "platformer_model.zip"

            pygame.draw.rect(screen, PANEL if is_sel else BG,
                             (ix, iy, iw, item_h), border_radius=6)
            pygame.draw.rect(screen, ACCENT if is_sel else BORDER,
                             (ix, iy, iw, item_h), 1, border_radius=6)
            if is_sel:
                pygame.draw.rect(screen, ACCENT, (ix, iy + 8, 3, item_h - 16))

            ls = font_body.render(label, True, TEXT_HI if is_sel else TEXT_MID)
            screen.blit(ls, (ix + 20, iy + item_h // 2 - ls.get_height() // 2))

            if is_final:
                tag = font_sm.render("FINAL", True, GOLD)
                screen.blit(tag, (ix + iw - tag.get_width() - 16,
                                  iy + item_h // 2 - tag.get_height() // 2))
            elif is_sel:
                tag = font_sm.render(os.path.basename(path), True, TEXT_LO)
                screen.blit(tag, (ix + iw - tag.get_width() - 16,
                                  iy + item_h // 2 - tag.get_height() // 2))

        # scroll hint
        if len(checkpoints) > VISIBLE:
            shown = font_sm.render(
                f"{scroll+1}–{min(scroll+VISIBLE, len(checkpoints))} of {len(checkpoints)}",
                True, TEXT_LO)
            screen.blit(shown, (SCREEN_W // 2 - shown.get_width() // 2,
                                sy + VISIBLE * (item_h + 8) + 8))

        foot = font_sm.render(
            "↑ ↓  select      ENTER  load      ESC  back", True, TEXT_LO)
        screen.blit(foot, (SCREEN_W // 2 - foot.get_width() // 2, SCREEN_H - 32))

        pygame.display.flip()


# ── AI WATCH ──────────────────────────────────────────────────────────────────
def run_ai_mode(screen, fonts):
    font_title, font_body, font_sm = fonts
    clock = pygame.time.Clock()

    # Pick checkpoint
    chosen = run_checkpoint_picker(screen, fonts)
    if chosen is None:
        return
    label, path = chosen

    # Loading screen
    draw_bg(screen)
    msg = font_body.render(f"Loading  {label} ...", True, TEXT_MID)
    screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))
    pygame.display.flip()

    model = load_model_from_path(path)
    if model is None:
        draw_bg(screen)
        err = font_body.render("Failed to load model — check console", True, DANGER)
        screen.blit(err, (SCREEN_W // 2 - err.get_width() // 2, SCREEN_H // 2))
        pygame.display.flip()
        time.sleep(2)
        return

    env       = PlatformerGauntletEnv(render_mode=None, seed=0)
    episode   = 0
    speed_idx = 0          # default 1×

    while True:
        episode += 1
        obs, _ = env.reset(seed=episode)
        done   = False

        while not done:
            # speed: 0=uncapped, else cap to FPS*multiplier
            spd = SPEEDS[speed_idx]
            if spd == 0:
                # uncapped — pump events but don't delay
                pygame.event.pump()
            else:
                clock.tick(FPS * spd)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close(); return
                    if event.key == pygame.K_1: speed_idx = 0
                    if event.key == pygame.K_2: speed_idx = 1
                    if event.key == pygame.K_3: speed_idx = 2
                    if event.key == pygame.K_4: speed_idx = 3

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # only render every frame at normal speed;
            # at high speed render every 3rd frame to keep it visible
            if spd <= 2 or episode % 3 == 0:
                cam_x = env.agent_x - SCREEN_W * 0.3
                draw_bg(screen)
                draw_level(screen, env.platforms, cam_x,
                           env.furthest_platform_idx)
                draw_agent(screen, env.agent_x - cam_x, env.agent_y)
                draw_progress_bar(screen, font_sm,
                                  env.furthest_platform_idx,
                                  len(env.platforms) - 1)
                draw_speed_indicator(screen, font_sm, speed_idx)
                draw_hint(screen, font_sm,
                          f"Episode {episode}   [{label}]   ESC → menu",
                          y=44)
                pygame.display.flip()

        time.sleep(0.3)


# ── HUMAN PLAY ────────────────────────────────────────────────────────────────
def run_human_mode(screen, fonts):
    _, font_body, font_sm = fonts
    clock     = pygame.time.Clock()
    best_ever = 0
    episode   = 0

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

        # death screen
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        lines = [
            (font_body, f"Reached platform {furthest} / {len(platforms)-1}", TEXT_HI),
            (font_sm,   f"Personal best:  {best_ever}",                      TEXT_MID),
            (font_sm,   "ENTER  retry       ESC  menu",                       TEXT_LO),
        ]
        total_h = sum(f.render(t, True, BG).get_height() + 10 for f, t, _ in lines)
        y = SCREEN_H // 2 - total_h // 2
        for fnt, txt, col in lines:
            s = fnt.render(txt, True, col)
            screen.blit(s, (SCREEN_W // 2 - s.get_width() // 2, y))
            y += s.get_height() + 10
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN: waiting = False
                    if event.key == pygame.K_ESCAPE: return


# ── TRAINING CHART ────────────────────────────────────────────────────────────
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
            msg = font_body.render(
                "No training_log.csv — run train.py first", True, DANGER)
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, SCREEN_H // 2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            clock.tick(FPS)
        return

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

        for tick in range(0, total + 1, 5):
            gy = PAD + int((1 - tick / total) * ch)
            pygame.draw.line(screen, PANEL, (PAD, gy), (PAD + cw, gy))
            lbl = font_sm.render(str(tick), True, TEXT_LO)
            screen.blit(lbl, (PAD - lbl.get_width() - 6, gy - 8))

        pygame.draw.line(screen, (*ACCENT, 80), (PAD, PAD), (PAD + cw, PAD), 1)

        n = len(vals)
        for i, v in enumerate(vals):
            px, py = to_px(i, v, n)
            pygame.draw.circle(screen, (*PLATFORM_C, 100), (px, py), 2)

        off = (n - len(smooth)) // 2
        pts = [to_px(off + i, v, n) for i, v in enumerate(smooth)]
        if len(pts) > 1:
            pygame.draw.lines(screen, GOLD, False, pts, 2)

        pygame.draw.rect(screen, BORDER, (PAD, PAD, cw, ch), 1)

        title = font_body.render("Skill Improvement over Training", True, TEXT_HI)
        screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 14))

        stats = (f"Episodes: {n}   Max platform: {max(vals)}"
                 f"   Rolling avg ({window} eps): {np.mean(vals[-window:]):.1f}")
        st = font_sm.render(stats, True, TEXT_MID)
        screen.blit(st, (PAD, SCREEN_H - 28))

        for i, (col, lbl) in enumerate([
                (GOLD,       "rolling average"),
                (PLATFORM_C, "per episode"),
                (ACCENT,     "full level")]):
            pygame.draw.rect(screen, col, (SCREEN_W - 190, 16 + i * 18, 14, 3))
            ls = font_sm.render(lbl, True, TEXT_MID)
            screen.blit(ls, (SCREEN_W - 172, 10 + i * 18))

        hint = font_sm.render("ESC  back to menu", True, TEXT_LO)
        screen.blit(hint, (SCREEN_W - hint.get_width() - 16, SCREEN_H - 28))
        pygame.display.flip()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Precision Platformer Gauntlet")

    font_title = pygame.font.SysFont("Arial", 28, bold=True)
    font_body  = pygame.font.SysFont("Arial", 20, bold=True)
    font_sm    = pygame.font.SysFont("Arial", 14)
    fonts      = (font_title, font_body, font_sm)

    has_model = bool(list_checkpoints())
    has_log   = os.path.exists("training_log.csv")

    while True:
        mode = run_menu(screen, fonts, has_model, has_log)
        if mode == "ai":
            run_ai_mode(screen, fonts)
        elif mode == "human":
            run_human_mode(screen, fonts)
        elif mode == "chart":
            run_chart_mode(screen, fonts)


if __name__ == "__main__":
    main()