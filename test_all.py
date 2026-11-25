from pathlib import Path

import cv2

from inspection import inspect_image

for p in sorted(Path("data/test_images").glob("*")):
    img = cv2.imread(str(p))
    if img is None:
        continue
    status, dets, _ = inspect_image(img)
    top = max((d["confidence"] for d in dets), default=0)
    classes = sorted({d["class"] for d in dets})
    print(f"{p.name:28s} {status:16s} top_conf={top:.2f}  {classes}")