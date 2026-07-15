"""Particle coalesce effect for the PHRONIS logo.

Random particles swirl in from the edges, converge to form the text
"PHRONIS", hold briefly, then the final frame is printed.
Rendered with braille characters for high detail.
"""

import math
import random
import time

from rich.console import Console
from rich.text import Text
from rich.align import Align
from rich.live import Live

from phronis.ui.braille import BrailleCanvas, text_to_pixels
from phronis.ui.timing import settle_curve, purple_from_white


class Particle:
    __slots__ = ("x", "y", "target_x", "target_y", "vx", "vy", "phase", "delay")

    def __init__(
        self, x: float, y: float, target_x: float, target_y: float, delay: float = 0
    ):
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.vx = 0.0
        self.vy = 0.0
        self.phase = random.uniform(0, math.pi * 2)
        self.delay = delay

    def update_converge(self, t: float, strength: float = 0.08, damping: float = 0.92):
        """Move toward target with spring-like physics."""
        if t < self.delay:
            self.x += self.vx
            self.y += self.vy
            self.vx *= 0.99
            self.vy *= 0.99
            angle = self.phase + t * 2
            self.vx += math.cos(angle) * 0.3
            self.vy += math.sin(angle) * 0.3
            return

        dx = self.target_x - self.x
        dy = self.target_y - self.y
        self.vx += dx * strength
        self.vy += dy * strength
        self.vx *= damping
        self.vy *= damping
        self.x += self.vx
        self.y += self.vy


def run_particle_logo(console: Console, hold_seconds: float = 2.0) -> None:
    """Run the particle coalesce effect."""
    term_width = min(console.width, 120)
    term_height = min(console.height - 4, 35)

    canvas = BrailleCanvas(term_width, term_height)

    text_pixels = text_to_pixels("PHRONIS", scale=2)

    def get_bounds(pixels):
        if not pixels:
            return 0, 0, 0, 0
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        return min(xs), max(xs), min(ys), max(ys)

    min_x, max_x, min_y, max_y = get_bounds(text_pixels)
    w, h = max_x - min_x + 1, max_y - min_y + 1

    offset_x = (canvas.pixel_width - w) // 2 - min_x
    offset_y = (canvas.pixel_height - h) // 2 - min_y
    targets = [(p[0] + offset_x, p[1] + offset_y) for p in text_pixels]

    step = max(1, len(targets) // 1500)
    sampled_targets = targets[::step]

    rng = random.Random(42)
    particles = []
    pw, ph = canvas.pixel_width, canvas.pixel_height

    for i, (tx, ty) in enumerate(sampled_targets):
        side = rng.choice(["top", "bottom", "left", "right"])
        if side == "top":
            sx, sy = rng.uniform(0, pw), rng.uniform(-20, -5)
        elif side == "bottom":
            sx, sy = rng.uniform(0, pw), rng.uniform(ph + 5, ph + 20)
        elif side == "left":
            sx, sy = rng.uniform(-20, -5), rng.uniform(0, ph)
        else:
            sx, sy = rng.uniform(pw + 5, pw + 20), rng.uniform(0, ph)

        delay = rng.uniform(0, 0.4)
        p = Particle(sx, sy, tx, ty, delay=delay)
        angle = math.atan2(ph / 2 - sy, pw / 2 - sx) + rng.gauss(0, 0.8)
        speed = rng.uniform(1.0, 2.5)
        p.vx = math.cos(angle) * speed
        p.vy = math.sin(angle) * speed
        particles.append(p)

    ambient = []
    for _ in range(200):
        ax = rng.uniform(0, pw)
        ay = rng.uniform(0, ph)
        ap = Particle(ax, ay, ax, ay)
        ap.vx = rng.gauss(0, 1)
        ap.vy = rng.gauss(0, 1)
        ambient.append(ap)

    fps = 24
    converge_frames = int(fps * 1.0)
    hold_frames = int(fps * hold_seconds)
    total_frames = converge_frames + hold_frames

    with Live(console=console, refresh_per_second=fps, transient=True) as live:
        for frame in range(total_frames):
            canvas.clear()
            t = frame * 0.03

            for ap in ambient:
                ap.x += ap.vx + math.sin(t + ap.phase) * 0.5
                ap.y += ap.vy + math.cos(t + ap.phase * 1.3) * 0.5
                ap.x = ap.x % pw
                ap.y = ap.y % ph

                if frame < converge_frames:
                    alpha = 0.3 + 0.2 * math.sin(t * 2 + ap.phase)
                else:
                    fade = (frame - converge_frames) / hold_frames
                    alpha = (0.3 + 0.2 * math.sin(t * 2 + ap.phase)) * (1 - fade)
                if alpha > 0.25:
                    canvas.set_pixel(int(ap.x), int(ap.y))

            if frame < converge_frames:
                progress = frame / converge_frames
                noise = settle_curve(progress)
                for p in particles:
                    p.update_converge(t, strength=0.06, damping=0.90)
                    canvas.set_pixel(int(p.x), int(p.y))

                    trail_scale = 0.2 + 0.5 * noise
                    trail_x = int(p.x - p.vx * trail_scale)
                    trail_y = int(p.y - p.vy * trail_scale)
                    canvas.set_pixel(trail_x, trail_y)

                r, g, b = purple_from_white(progress)
            else:
                settle_t = (frame - converge_frames) / hold_frames
                for p in particles:
                    jitter = (1 - settle_t) * 0.7
                    jx = p.target_x + math.sin(t * 3 + p.phase) * jitter
                    jy = p.target_y + math.cos(t * 3 + p.phase * 1.5) * jitter
                    canvas.set_pixel(int(jx), int(jy))
                    canvas.set_pixel(int(p.target_x), int(p.target_y))

                r, g, b = 168, 85, 247

            lines = canvas.render()
            result = Text()
            for line in lines:
                for ch in line:
                    if ch == chr(0x2800):
                        result.append(ch)
                    else:
                        result.append(ch, style=f"rgb({r},{g},{b})")
                result.append("\n")

            live.update(Align.center(result))
            time.sleep(1.0 / fps)

    canvas.clear()
    for p in particles:
        canvas.set_pixel(int(p.target_x), int(p.target_y))
    final = Text()
    for line in canvas.render():
        for ch in line:
            if ch == chr(0x2800):
                final.append(ch)
            else:
                final.append(ch, style="rgb(168,85,247)")
        final.append("\n")
    console.print(Align.center(final))
