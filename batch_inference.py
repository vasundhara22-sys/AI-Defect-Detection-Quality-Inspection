import csv
import os
import sys
from collections import Counter
from pathlib import Path

import requests

API = os.environ.get("DEFECT_API", "http://127.0.0.1:8000")
folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/test_images")
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
files = sorted(p for p in folder.glob("*") if p.suffix.lower() in EXT)
if not files:
    sys.exit(f"No images found in {folder}")


def truth_of(name):
    # Files starting with "good" are defect-free. Everything else is defective.
    return "good" if name.lower().startswith("good") else "defect"


rows = []
for p in files:
    with open(p, "rb") as f:
        r = requests.post(f"{API}/inspect",
                          files={"file": (p.name, f, "image/jpeg")},
                          data={"product_id": p.stem}, timeout=120)
    if not r.ok:
        print("FAILED", p.name, r.text)
        continue
    res = r.json()
    dets = res["detections"]
    top = max((d["confidence"] for d in dets), default=0)
    found = sorted({d["class"] for d in dets})
    rows.append([p.name, truth_of(p.name), res["status"], round(top, 3), " ".join(found)])

Path("reports").mkdir(exist_ok=True)
with open("reports/batch_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "truth", "status", "top_confidence", "classes_found"])
    w.writerows(rows)

defects = [r for r in rows if r[1] == "defect"]
goods = [r for r in rows if r[1] == "good"]
missed = [r for r in defects if r[2] == "Pass"]
false_alarms = [r for r in goods if r[2] != "Pass"]

print(f"Defective images: {len(defects)}   missed (Pass): {len(missed)}")
if defects:
    print(f"Missed-defect rate: {100 * len(missed) / len(defects):.0f}%")
print(f"Good images: {len(goods)}   false alarms (Fail or Review): {len(false_alarms)}")
if goods:
    print(f"False-alarm rate: {100 * len(false_alarms) / len(goods):.0f}%")
print("Status counts:", dict(Counter(r[2] for r in rows)))
print("Saved: reports/batch_results.csv")