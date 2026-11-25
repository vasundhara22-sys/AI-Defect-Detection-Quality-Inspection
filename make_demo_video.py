from pathlib import Path

import cv2

imgs = sorted(Path("data/test_images").glob("*"))
out = cv2.VideoWriter("data/demo_conveyor.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 640))

for p in imgs:
    frame = cv2.imread(str(p))
    if frame is None:
        continue
    frame = cv2.resize(frame, (640, 640))
    cv2.putText(frame, p.stem, (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for _ in range(20):
        out.write(frame)
out.release()
print("Saved data/demo_conveyor.mp4 with", len(imgs), "products")