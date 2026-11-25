from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("data/my_product_test")
OUT.mkdir(parents=True, exist_ok=True)
S = 400


def draw_bolt(seed, scratched):
    rng = np.random.default_rng(seed)
    y = np.arange(S)[:, None] * np.ones((1, S))
    bg = (235 - 25 * (y / S)).astype(np.uint8)
    im = Image.fromarray(np.stack([bg, bg, bg], axis=-1))
    d = ImageDraw.Draw(im)
    cx = 200 + int(rng.integers(-15, 16))
    hexpts = [(cx + 70 * np.cos(a), 110 + 70 * np.sin(a)) for a in np.arange(0, 2 * np.pi, np.pi / 3)]
    d.polygon(hexpts, fill=(150, 150, 155))
    d.rectangle([cx - 28, 145, cx + 28, 335], fill=(165, 165, 170))
    if scratched:
        for _ in range(int(rng.integers(2, 4))):
            x0 = int(cx + rng.integers(-18, 19))
            y0 = int(rng.integers(170, 310))
            L = int(rng.integers(30, 60))
            d.line([(x0, y0), (x0 + int(rng.integers(-8, 9)), y0 + L)], fill=(40, 40, 45), width=2)
    arr = np.asarray(im.filter(ImageFilter.GaussianBlur(0.4))).astype(np.float32)
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


for i in range(10):
    draw_bolt(700 + i, False).save(OUT / f"good_{i + 1:02d}.jpg", quality=92)
    draw_bolt(800 + i, True).save(OUT / f"scratch_{i + 1:02d}.jpg", quality=92)
print("Wrote 20 images to", OUT)
