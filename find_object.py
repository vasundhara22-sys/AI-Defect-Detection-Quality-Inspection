import sys
from pathlib import Path

import cv2
from ultralytics import YOLOWorld

# usage: python find_object.py <image> "name1" "name2" ...
img_path = sys.argv[1]
names = sys.argv[2:]

model = YOLOWorld("yolov8s-world.pt")   # downloads once, needs internet
model.set_classes(names)
res = model.predict(img_path, conf=0.1, verbose=False)[0]

img = cv2.imread(img_path)
for b in res.boxes:
    cls = names[int(b.cls)]
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    conf = float(b.conf)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{cls} {conf:.2f}", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print(cls, round(conf, 3), (x1, y1, x2, y2))

cv2.imwrite("find_result.jpg", img)
print("Saved find_result.jpg with", len(res.boxes), "detections")
