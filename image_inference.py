import sys

import cv2

from inspection import inspect_image

path = sys.argv[1]
image = cv2.imread(path)
if image is None:
    raise SystemExit(f"ERROR: could not read image: {path}")

status, detections, annotated = inspect_image(image)

print("Result:", status)
for d in detections:
    print("  ", d)
cv2.imwrite("reports/result.jpg", annotated)
print("Saved reports/result.jpg")