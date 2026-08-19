"""
launcher.py — Precision Platformer Gauntlet
Modes: Watch AI (with checkpoint picker + speed control) | Play | Chart
"""

import os, sys, time
import pygame
import numpy as np

from platformer_env import (
    PlatformerGauntletEnv, SCREEN_W, SCREEN_H,
    AGENT_W, AGENT_H, GRAVITY, JUMP_VELOCITY,
    MOVE_SPEED, TERMINAL_VY, PLATFORM_THICKNESS, generate_level,
)

BG         = (12,  14,  23)
PANEL      = (20,  24,  40)
ACCENT     = (99, 210, 140)
PLATFORM_C = (75,  95, 180)
AGENT_COL  = (240, 200,  60)
TEXT_HI    = (230, 230, 230)
TEXT_MID   = (150, 155, 170)
TEXT_LO    = (110, 120, 150)
DANGER     = (220,  80,  80)
GOLD       = (255, 200,  60)
FPS        = 60
SPEEDS     = [1, 2, 5, 0]
SPEED_LBLS = ["1×", "2×", "5×", "MAX"]


# ── model helpers ─────────────────────────────────────────────────────────────
def list_checkpoints():
    items = []
    ckpt = "checkpoints"
    if os.path.isdir(ckpt):
        zips = sorted(
            [f for f in os.listdir(ckpt) if f.endswith(".zip")],
            key=lambda f: int("".join(filter(str.isdigit, f)) or 0))
        for z in zips:
            steps = int("".join(filter(str.isdigit, z)) or 0)
            items.append((f"{steps:,} steps", os.path.join(ckpt, z)))
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


# ── collision (AABB — fixes human mode phasing) ───────────────────────────────
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
                return True, p.y - AGENT_H, 0.0, idx
    return False, ay, vy, cur_idx


# ── shared draw ───────────────────────────────────────────────────────────────
def draw_level(surface, platforms, cam_x, furthest_idx):
    for idx, p in enumerate(platforms):
        sx = p.x - cam_x
        if sx + p.width < 0 or sx > SCREEN_W:
            continue
        col = ACCENT if idx <= furthest_idx else PLATFORM_C
        pygame.draw.rect(surface, col, (int(sx), int(p.y), int(p.width), PLATFORM_THICKNESS))


def draw_agent(surface, ax, ay):
    pygame.draw.rect(surface, AGENT_COL, (int(ax), int(ay), AGENT_W, AGENT_H))


def draw_bar(surface, font, furthest, total):
    bw, bh = 260, 10
    bx, by = SCREEN_W - bw - 16, 14
    pct = furthest / max(total, 1)
    pygame.draw.rect(surface, PANEL,  (bx, by, bw, bh), border_radius=4)
    if pct > 0:
        pygame.draw.rect(surface, ACCENT, (bx, by, int(bw * pct), bh), border_radius=4)
    pygame.draw.rect(surface, (50, 58, 90), (bx, by, bw, bh), 1, border_radius=4)
    lbl = font.render(f"Platform  {furthest} / {total}", True, TEXT_MID)
    surface.blit(lbl, (bx + bw - lbl.get_width(), by + bh + 5))


def draw_speed_bar(surface, font, speed_idx):
    bw, bh, gap = 50, 22, 5
    total_w = len(SPEEDS) * bw + (len(SPEEDS) - 1) * gap
    sx = SCREEN_W // 2 - total_w // 2
    sy = 12
    for i, lbl in enumerate(SPEED_LBLS):
        bx = sx + i * (bw + gap)
        active = i == speed_idx
        pygame.draw.rect(surface, ACCENT if active else PANEL, (bx, sy, bw, bh), border_radius=4)
        pygame.draw.rect(surface, (50, 58, 90), (bx, sy, bw, bh), 1, border_radius=4)
        t = font.render(lbl, True, BG if active else TEXT_LO)
        surface.blit(t, (bx + bw//2 - t.get_width()//2, sy + bh//2 - t.get_height()//2))
    hint = font.render("1 2 3 4  speed", True, TEXT_LO)
    surface.blit(hint, (sx + total_w + 8, sy + 4))


def draw_hint(surface, font, text, y=16):
    surface.blit(font.render(text, True, TEXT_LO), (16, y))


# ── MENU ──────────────────────────────────────────────────────────────────────
def run_menu(screen, fonts):
    font_title, font_body, font_sm = fonts
    clock = pygame.time.Clock()
    has_model = bool(list_checkpoints())
    has_log = os.path.exists("training_log.csv")

    options = [
        ("Watch AI Play",  has_model, "ai"),
        ("Play Yourself",  True,      "human"),
        ("Training Chart", has_log,   "chart"),
    ]
    sel = 0

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    sel = (sel - 1) % len(options)
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    sel = (sel + 1) % len(options)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    if options[sel][1]:
                        return options[sel][2]
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

        screen.fill(BG)
        title = font_title.render("PRECISION PLATFORMER GAUNTLET", True, TEXT_HI)
        sub   = font_sm.render("Neural Network Learning Demo", True, TEXT_MID)
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 72))
        screen.blit(sub,   (SCREEN_W//2 - sub.get_width()//2, 72 + title.get_height() + 6))

        div_y = 72 + title.get_height() + sub.get_height() + 22
        pygame.draw.line(screen, (35, 42, 65), (SCREEN_W//4, div_y), (SCREEN_W*3//4, div_y))

        iw, ih = 420, 64
        sy = div_y + 36
        for i, (name, enabled, _) in enumerate(options):
            iy = sy + i * (ih + 12)
            ix = SCREEN_W//2 - iw//2
            is_sel = i == sel
            pygame.draw.rect(screen, PANEL if is_sel else BG, (ix, iy, iw, ih), border_radius=6)
            pygame.draw.rect(screen, ACCENT if is_sel else (35, 42, 65), (ix, iy, iw, ih), 1, border_radius=6)
            if is_sel:
                pygame.draw.rect(screen, ACCENT, (ix, iy+10, 3, ih-20))
            col = TEXT_HI if enabled else TEXT_LO
            ns = font_body.render(name, True, col)
            screen.blit(ns, (ix + 20, iy + ih//2 - ns.get_height()//2))
            if not enabled:
                ws = font_sm.render("not available", True, DANGER)
                screen.blit(ws, (ix + iw - ws.get_width() - 16, iy + ih//2 - ws.get_height()//2))

        foot = font_sm.render("↑ ↓  navigate      ENTER  select      ESC  quit", True, TEXT_LO)
        screen.blit(foot, (SCREEN_W//2 - foot.get_width()//2, SCREEN_H - 30))
        pygame.display.flip()


# ── CHECKPOINT PICKER ─────────────────────────────────────────────────────────
def run_checkpoint_picker(screen, fonts):
    font_title, font_body, font_sm = fonts
    clock = pygame.time.Clock()
    items = list_checkpoints()
    if not items:
        return None

    sel = len(items) - 1
    scroll = 0
    VIS = 8
    ih = 52

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    sel = max(0, sel - 1)
                    scroll = min(scroll, sel)
                if event.key in (pygame.K_DOWN, pygame.K_s):
                    sel = min(len(items) - 1, sel + 1)
                    scroll = max(scroll, sel - VIS + 1)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    return items[sel]
                if event.key == pygame.K_ESCAPE:
                    return None

        screen.fill(BG)
        title = font_title.render("SELECT CHECKPOINT", True, TEXT_HI)
        sub   = font_sm.render("Choose training snapshot — ENTER to load", True, TEXT_MID)
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 56))
        screen.blit(sub,   (SCREEN_W//2 - sub.get_width()//2, 56 + title.get_height() + 6))

        iw, ix = 480, SCREEN_W//2 - 240
        sy = 140
        for i, (label, path) in enumerate(items[scroll:scroll+VIS]):
            real_idx = scroll + i
            iy = sy + i * (ih + 8)
            is_sel = real_idx == sel
            is_final = path == "platformer_model.zip"
            pygame.draw.rect(screen, PANEL if is_sel else BG, (ix, iy, iw, ih), border_radius=6)
            pygame.draw.rect(screen, ACCENT if is_sel else (35,42,65), (ix, iy, iw, ih), 1, border_radius=6)
            if is_sel:
                pygame.draw.rect(screen, ACCENT, (ix, iy+8, 3, ih-16))
            ls = font_body.render(label, True, TEXT_HI if is_sel else TEXT_MID)
            screen.blit(ls, (ix+20, iy + ih//2 - ls.get_height()//2))
            if is_final:
                tag = font_sm.render("FINAL", True, GOLD)
                screen.blit(tag, (ix + iw - tag.get_width() - 16, iy + ih//2 - tag.get_height()//2))

        foot = font_sm.render("↑ ↓  select      ENTER  load      ESC  back", True, TEXT_LO)
        screen.blit(foot, (SCREEN_W//2 - foot.get_width()//2, SCREEN_H - 30))
        pygame.display.flip()


# ── AI WATCH ──────────────────────────────────────────────────────────────────
def run_ai_mode(screen, fonts):
    _, font_body, font_sm = fonts
    clock = pygame.time.Clock()

    chosen = run_checkpoint_picker(screen, fonts)
    if chosen is None:
        return
    label, path = chosen

    screen.fill(BG)
    msg = font_body.render(f"Loading {label}...", True, TEXT_MID)
    screen.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2))
    pygame.display.flip()

    model = load_model(path)
    if model is None:
        return

    env = PlatformerGauntletEnv(render_mode=None, seed=0)
    episode = 0
    speed_idx = 0

    while True:
        episode += 1
        obs, _ = env.reset(seed=episode)
        done = False

        while not done:
            spd = SPEEDS[speed_idx]
            if spd:
                clock.tick(FPS * spd)
            else:
                pygame.event.pump()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close(); pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close(); return
                    for k, idx in [(pygame.K_1,0),(pygame.K_2,1),(pygame.K_3,2),(pygame.K_4,3)]:
                        if event.key == k:
                            speed_idx = idx

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if spd <= 2 or episode % 3 == 0:
                cam_x = env.agent_x - SCREEN_W * 0.3
                screen.fill(BG)
                draw_level(screen, env.platforms, cam_x, env.furthest_platform_idx)
                draw_agent(screen, env.agent_x - cam_x, env.agent_y)
                draw_bar(screen, font_sm, env.furthest_platform_idx, len(env.platforms)-1)
                draw_speed_bar(screen, font_sm, speed_idx)
                draw_hint(screen, font_sm, f"Episode {episode}   [{label}]   ESC → menu", y=44)
                pygame.display.flip()

        time.sleep(0.3)


# ── HUMAN PLAY ────────────────────────────────────────────────────────────────
def run_human_mode(screen, fonts):
    _, font_body, font_sm = fonts
    clock = pygame.time.Clock()
    best_ever = 0
    episode = 0

    while True:
        episode += 1
        platforms = generate_level(seed=0)
        start = platforms[0]
        ax = start.x + start.width / 2 - AGENT_W / 2
        ay = start.y - AGENT_H
        vx = vy = 0.0
        prev_ay = ay
        on_ground = True
        furthest = 0
        cur_idx = 0
        t = 0

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
            vx = (-MOVE_SPEED if keys[pygame.K_LEFT] else
                   MOVE_SPEED if keys[pygame.K_RIGHT] else 0.0)
            if (keys[pygame.K_SPACE] or keys[pygame.K_UP]) and on_ground:
                vy = JUMP_VELOCITY
                on_ground = False

            vy = min(vy + GRAVITY, TERMINAL_VY)
            prev_ay = ay
            ax += vx
            ay += vy

            for p in platforms:
                p.update(t)

            landed, ay, vy, new_idx = check_landing(ax, ay, prev_ay, vy, platforms, cur_idx)
            if landed:
                on_ground = True
                cur_idx = new_idx
                if new_idx > furthest:
                    furthest = new_idx
                    best_ever = max(best_ever, furthest)
            else:
                on_ground = False

            cam_x = ax - SCREEN_W * 0.3
            screen.fill(BG)
            draw_level(screen, platforms, cam_x, furthest)
            draw_agent(screen, ax - cam_x, ay)
            draw_bar(screen, font_sm, furthest, len(platforms)-1)
            draw_hint(screen, font_sm, f"← → move   SPACE jump   ESC menu   Best: {best_ever}")
            pygame.display.flip()

            if ay > SCREEN_H + 100:
                running = False

        # death screen
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        for txt, col, dy in [
            (font_body.render(f"Reached platform {furthest} / {len(platforms)-1}", True, TEXT_HI), TEXT_HI, -40),
            (font_sm.render(f"Personal best: {best_ever}", True, TEXT_MID), TEXT_MID, 0),
            (font_sm.render("ENTER retry   ESC menu", True, TEXT_LO), TEXT_LO, 36),
        ]:
            screen.blit(txt, (SCREEN_W//2 - txt.get_width()//2, SCREEN_H//2 + dy))
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
            rows = list(csv.DictReader(f))
            vals = [int(r["furthest_platform"]) for r in rows]
            total = int(rows[0]["total_platforms"]) if rows else 40
    except Exception:
        vals, total = [], 40

    if not vals:
        while True:
            screen.fill(BG)
            msg = font_body.render("No training_log.csv — run train.py first", True, DANGER)
            screen.blit(msg, (SCREEN_W//2 - msg.get_width()//2, SCREEN_H//2))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return
            clock.tick(FPS)
        return

    window = max(1, len(vals) // 20)
    smooth = np.convolve(vals, np.ones(window)/window, mode="valid")
    PAD = 64
    cw = SCREEN_W - PAD*2
    ch = SCREEN_H - PAD*2 - 48

    def to_px(i, v, n):
        return (PAD + int(i/max(n-1,1)*cw),
                PAD + int((1 - v/max(total,1))*ch))

    while True:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return

        screen.fill(BG)
        for tick in range(0, total+1, 5):
            gy = PAD + int((1 - tick/total)*ch)
            pygame.draw.line(screen, PANEL, (PAD, gy), (PAD+cw, gy))
            screen.blit(font_sm.render(str(tick), True, TEXT_LO), (PAD-30, gy-8))

        n = len(vals)
        for i, v in enumerate(vals):
            pygame.draw.circle(screen, (*PLATFORM_C, 100), to_px(i, v, n), 2)

        off = (n - len(smooth)) // 2
        pts = [to_px(off+i, v, n) for i, v in enumerate(smooth)]
        if len(pts) > 1:
            pygame.draw.lines(screen, GOLD, False, pts, 2)

        pygame.draw.rect(screen, (35,42,65), (PAD, PAD, cw, ch), 1)
        title = font_body.render("Skill Improvement over Training", True, TEXT_HI)
        screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 14))
        stats = f"Episodes: {n}   Max: {max(vals)}   Rolling avg ({window}): {np.mean(vals[-window:]):.1f}"
        screen.blit(font_sm.render(stats, True, TEXT_MID), (PAD, SCREEN_H-28))
        hint = font_sm.render("ESC  back to menu", True, TEXT_LO)
        screen.blit(hint, (SCREEN_W - hint.get_width() - 16, SCREEN_H-28))
        pygame.display.flip()


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Precision Platformer Gauntlet")
    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20, bold=True),
        pygame.font.SysFont("Arial", 14),
    )
    while True:
        mode = run_menu(screen, fonts)
        if mode == "ai":     run_ai_mode(screen, fonts)
        elif mode == "human": run_human_mode(screen, fonts)
        elif mode == "chart": run_chart_mode(screen, fonts)


if __name__ == "__main__":
    main()
