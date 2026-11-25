import csv
import sys
from datetime import datetime

import cv2

import config
from inspection import decide, inspect_image

source = sys.argv[1] if len(sys.argv) > 1 else "data/demo_conveyor.mp4"   # "0" = webcam
CHANGE_THRESHOLD = 10    # mean pixel difference that signals a new product
EVERY_N = 3              # run the model on every 3rd frame (faster on CPU)

cap = cv2.VideoCapture(0 if source == "0" else source)
if not cap.isOpened():
    raise SystemExit(f"ERROR: cannot open video: {source}")
w, h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("reports/output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))

COLOURS = {"Pass": (0, 200, 0), "Fail": (0, 0, 255), "Review Required": (0, 165, 255)}
products = []            # finished products: dicts
current = {"confs": [], "classes": set()}
prev_small, last_view, frame_no = None, None, 0


def finish_product():
    if not current["confs"] and not products and frame_no == 0:
        return
    status = decide(current["confs"])
    products.append({
        "product_no": len(products) + 1,
        "status": status,
        "top_confidence": round(max(current["confs"], default=0), 3),
        "defects": ";".join(sorted(current["classes"])),
        "time": datetime.now().isoformat(timespec="seconds"),
    })


while True:
    ok, frame = cap.read()
    if not ok:
        break
    small = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
    if prev_small is not None and cv2.absdiff(small, prev_small).mean() > CHANGE_THRESHOLD:
        finish_product()                       # previous product is complete
        current = {"confs": [], "classes": set()}
    prev_small = small

    if frame_no % EVERY_N == 0:
        status, dets, annotated = inspect_image(frame)
        for d in dets:
            current["confs"].append(d["confidence"])
            current["classes"].add(d["class"])
        last_view = annotated
    view = (last_view if last_view is not None else frame).copy()

    live = decide(current["confs"])
    cv2.putText(view, f"Product {len(products) + 1}: {live}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOURS[live], 3)
    out.write(view)
    cv2.imshow("Inspection (press q to quit)", view)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_no += 1

finish_product()                               # last product
cap.release(); out.release(); cv2.destroyAllWindows()

with open("reports/video_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["product_no", "status", "top_confidence", "defects", "time"])
    writer.writeheader()
    writer.writerows(products)

counts = {}
for p in products:
    counts[p["status"]] = counts.get(p["status"], 0) + 1
print("Products inspected:", len(products))
print("Summary:", counts)
print("Saved reports/video_results.csv and reports/output_video.mp4")