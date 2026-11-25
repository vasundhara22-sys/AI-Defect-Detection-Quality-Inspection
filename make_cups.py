from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("data/cups")
OUT.mkdir(parents=True, exist_ok=True)
S = 400


def draw_cup(seed, scratches):
    rng = np.random.default_rng(seed)
    y = np.arange(S)[:, None] * np.ones((1, S))
    bg = (215 - 40 * (y / S)).astype(np.uint8)
    im = Image.fromarray(np.stack([bg, bg, bg], axis=-1))
    d = ImageDraw.Draw(im)
    top_l, top_r, bot_l, bot_r = 105, 295, 120, 280
    top_y, bot_y = 110, 330
    d.polygon([(top_l, top_y), (top_r, top_y), (bot_r, bot_y), (bot_l, bot_y)], fill=(236, 236, 232))
    d.arc([265, 150, 345, 260], -90, 90, fill=(220, 220, 216), width=14)
    d.ellipse([top_l, top_y - 14, top_r, top_y + 14], fill=(200, 200, 196))
    d.ellipse([top_l + 8, top_y - 9, top_r - 8, top_y + 9], fill=(120, 110, 100))
    d.ellipse([bot_l - 10, bot_y - 8, bot_r + 10, bot_y + 14], fill=(170, 170, 166))
    if scratches:
        for _ in range(int(rng.integers(3, 6))):
            x0 = int(rng.integers(125, 265))
            y0 = int(rng.integers(130, 300))
            length = int(rng.integers(25, 60))
            a = float(rng.uniform(-0.6, 0.6))
            x1 = int(x0 + length * np.cos(a))
            y1 = int(y0 + length * np.sin(a))
            d.line([(x0, y0), (x1, y1)], fill=(150, 145, 140), width=1)
            d.line([(x0 + 1, y0 + 1), (x1 + 1, y1 + 1)], fill=(90, 85, 80), width=1)
    return im.filter(ImageFilter.GaussianBlur(0.5))


for i in range(6):
    draw_cup(100 + i, True).save(OUT / f"cup_scratched_{i + 1:02d}.jpg", quality=92)
for i in range(4):
    draw_cup(200 + i, False).save(OUT / f"good_cup_clean_{i + 1:02d}.jpg", quality=92)
print("Wrote 10 images to", OUT)