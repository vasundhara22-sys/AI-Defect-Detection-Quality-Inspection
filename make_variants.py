from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/test_images")
DST = Path("data/robust_images")
DST.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)


def variants(img):
    h, w = img.shape[:2]
    noise = rng.normal(0, 12, img.shape)
    canvas = np.full((h * 2, w * 2, 3), 90, np.uint8)       # product small in a larger frame
    canvas[h // 2:h // 2 + h, w // 2:w // 2 + w] = img
    return {
        "dark": cv2.convertScaleAbs(img, alpha=0.6, beta=-10),
        "bright": cv2.convertScaleAbs(img, alpha=1.0, beta=50),
        "blur": cv2.GaussianBlur(img, (7, 7), 0),
        "noise": np.clip(img + noise, 0, 255).astype(np.uint8),
        "rot90": cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        "flip": cv2.flip(img, 1),
        "padded": canvas,
    }


count = 0
for p in sorted(SRC.glob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    for name, v in variants(img).items():
        cv2.imwrite(str(DST / f"{p.stem}-{name}.jpg"), v)
        count += 1
    cv2.imwrite(str(DST / f"{p.stem}-jpeg25.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 25])
    count += 1
print(f"Wrote {count} images to {DST}")