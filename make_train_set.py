from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("data/real_train")
OUT.mkdir(parents=True, exist_ok=True)
S = 400


def bolt(seed, kind):
    rng = np.random.default_rng(seed)
    y = np.arange(S)[:, None] * np.ones((1, S))
    bg = (235 - 25 * (y / S)).astype(np.uint8)
    im = Image.fromarray(np.stack([bg, bg, bg], axis=-1))
    d = ImageDraw.Draw(im)
    cx = 200 + int(rng.integers(-15, 16))
    hexpts = [(cx + 70 * np.cos(a), 110 + 70 * np.sin(a)) for a in np.arange(0, 2 * np.pi, np.pi / 3)]
    d.polygon(hexpts, fill=(150, 150, 155))
    boxes = []
    if kind == "broken":
        gap = int(rng.integers(200, 260))
        d.rectangle([cx - 28, 145, cx + 28, gap], fill=(165, 165, 170))
        d.rectangle([cx - 28, gap + 30, cx + 28, 335], fill=(165, 165, 170))
        boxes.append(("broken", cx - 40, gap - 5, cx + 40, gap + 35))
    else:
        d.rectangle([cx - 28, 145, cx + 28, 335], fill=(165, 165, 170))
    if kind == "scratch":
        for _ in range(int(rng.integers(2, 4))):
            x0 = int(cx + rng.integers(-18, 19))
            y0 = int(rng.integers(170, 310))
            x1 = x0 + int(rng.integers(-8, 9))
            y1 = y0 + int(rng.integers(30, 60))
            d.line([(x0, y0), (x1, y1)], fill=(40, 40, 45), width=2)
            boxes.append(("scratch", min(x0, x1) - 6, y0 - 4, max(x0, x1) + 6, y1 + 4))
    arr = np.asarray(im.filter(ImageFilter.GaussianBlur(0.4))).astype(np.float32)
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), boxes


def write_xml(path, boxes):
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{max(0, x1)}</xmin><ymin>{max(0, y1)}</ymin>"
        f"<xmax>{min(S, x2)}</xmax><ymax>{min(S, y2)}</ymax></bndbox></object>"
        for n, x1, y1, x2, y2 in boxes)
    path.write_text(f"<annotation><size><width>{S}</width><height>{S}</height><depth>3</depth></size>{objs}</annotation>")


for kind, seed in [("good", 900), ("scratch", 910), ("broken", 920)]:
    for i in range(10):
        im, boxes = bolt(seed + i, kind)
        stem = f"{kind}_{i + 1:02d}"
        im.save(OUT / f"{stem}.jpg", quality=92)
        write_xml(OUT / f"{stem}.xml", boxes)
print("Wrote 30 images and labels to", OUT)
