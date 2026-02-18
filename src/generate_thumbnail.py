"""
generate_thumbnail.py
Creates YouTube thumbnail with smart text wrapping.
"""

import math
import random
import logging
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

log = logging.getLogger("yt-uploader")

THUMB_W, THUMB_H = 1280, 720

GRADIENT_SETS = [
    [(30, 0, 80),   (120, 0, 180),  (255, 100, 255)],
    [(0, 20, 80),   (0, 100, 200),  (0, 220, 255)],
    [(60, 0, 0),    (180, 40, 0),   (255, 160, 0)],
    [(0, 40, 30),   (0, 120, 80),   (80, 255, 180)],
]


def generate_thumbnail(song_title: str, artist: str,
                       channel_name: str, output_dir: str) -> str:
    palette      = random.choice(GRADIENT_SETS)
    c1, c2, accent = palette

    img  = Image.new("RGB", (THUMB_W, THUMB_H))
    draw = ImageDraw.Draw(img)

    # Gradient
    for y in range(THUMB_H):
        for x in range(THUMB_W):
            t = x / THUMB_W * 0.5 + y / THUMB_H * 0.5
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.point((x, y), fill=(r, g, b))

    # Glowing orb
    orb_x, orb_y = THUMB_W // 3, THUMB_H // 2
    for r in range(300, 0, -10):
        color = tuple(min(255, c + 80) for c in accent)
        draw.ellipse([orb_x - r, orb_y - r, orb_x + r, orb_y + r], fill=color)

    # Geometric lines
    for i in range(8):
        angle  = math.radians(i * 22.5)
        x1 = orb_x + math.cos(angle) * 50
        y1 = orb_y + math.sin(angle) * 50
        x2 = orb_x + math.cos(angle) * 400
        y2 = orb_y + math.sin(angle) * 400
        draw.line([(x1, y1), (x2, y2)], fill=accent, width=2)

    # Waveform
    _draw_waveform(draw, accent)

    # Dark right panel
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)
    od.rectangle([THUMB_W // 2, 0, THUMB_W, THUMB_H], fill=(0, 0, 0, 160))
    img  = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Vignette
    img  = _vignette(img)
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        font_title   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
        font_artist  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        font_badge   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        font_channel = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except Exception:
        font_title = font_artist = font_badge = font_channel = ImageFont.load_default()

    text_cx = THUMB_W * 3 // 4

    def centered(text, font, y, color=(255, 255, 255)):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
        except Exception:
            tw = len(text) * 20
        x = text_cx - tw // 2
        draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=color)

    # Badge
    badge   = "◆  SLOWED + REVERB  ◆"
    badge_y = 70
    try:
        bbox    = draw.textbbox((0, 0), badge, font=font_badge)
        bw, bh  = bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 20
    except Exception:
        bw, bh = 400, 60
    bx = text_cx - bw // 2
    draw.rounded_rectangle([bx, badge_y, bx + bw, badge_y + bh],
                            radius=15, fill=tuple(min(255, c + 30) for c in accent))
    centered(badge, font_badge, badge_y + 8, color=(0, 0, 0))

    # Song title - wrap if needed
    title_lines = _wrap_text(song_title, font_title, 500, draw)
    y = 170
    for line in title_lines:
        centered(line, font_title, y)
        y += 80

    # Artist
    y += 20
    artist_display = artist if len(artist) <= 25 else artist[:23] + "…"
    centered(f"— {artist_display} —", font_artist, y, color=(200, 220, 255))

    # Divider
    y += 70
    draw.line([text_cx - 250, y, text_cx + 250, y], fill=(255, 255, 255, 100), width=2)

    # Channel
    y += 15
    centered(f"♫  {channel_name}", font_channel, y, color=(180, 180, 255))

    # Border
    draw.rectangle([0, 0, THUMB_W - 1, THUMB_H - 1], outline=accent, width=6)

    out_path = Path(output_dir) / f"{_safe(song_title)}_thumbnail.jpg"
    img.save(str(out_path), "JPEG", quality=95, optimize=True)
    log.info(f"  ✓ Thumbnail: {out_path.name}")
    return str(out_path)


def _wrap_text(text: str, font, max_width: int, draw) -> list:
    """Wrap text into multiple lines if needed."""
    try:
        bbox   = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
    except Exception:
        text_w = len(text) * 35

    if text_w <= max_width:
        return [text]

    words = text.split()
    lines = []
    current = []

    for word in words:
        test = " ".join(current + [word])
        try:
            bbox   = draw.textbbox((0, 0), test, font=font)
            test_w = bbox[2] - bbox[0]
        except Exception:
            test_w = len(test) * 35

        if test_w <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    # Max 2 lines
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1][:25] + "…"

    return lines


def _draw_waveform(draw, accent):
    bars    = 40
    bar_w   = 8
    spacing = 14
    start_x = 80
    max_h   = 200
    cy      = THUMB_H // 2
    for i in range(bars):
        height = int(max_h * abs(math.sin(i * 0.35)) * random.uniform(0.5, 1.0))
        x      = start_x + i * (bar_w + spacing)
        color  = tuple(min(255, c + 60) for c in accent)
        draw.rectangle([x, cy - height // 2, x + bar_w, cy + height // 2], fill=color)


def _vignette(img: Image.Image) -> Image.Image:
    vignette = Image.new("L", (THUMB_W, THUMB_H), 0)
    v        = ImageDraw.Draw(vignette)
    steps    = 40
    max_m    = min(THUMB_W, THUMB_H) // 2 - 1

    for i in range(steps):
        f      = i / steps
        margin = int((1 - f) * max_m)
        x0     = max(0, margin)
        y0     = max(0, margin)
        x1     = max(x0 + 1, THUMB_W - margin)
        y1     = max(y0 + 1, THUMB_H - margin)
        v.rectangle([x0, y0, x1, y1], fill=int(255 * f))

    vignette = vignette.filter(ImageFilter.GaussianBlur(60))
    black    = Image.new("RGB", (THUMB_W, THUMB_H), (0, 0, 0))
    return Image.composite(img, black, vignette)


def _safe(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:40]
