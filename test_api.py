from pathlib import Path

import requests

API = "http://127.0.0.1:8000"

# 1. Upload up to 6 test images, one by one
images = sorted(Path("data/test_images").glob("*"))[::4][:6]
for i, p in enumerate(images, start=1):
    with open(p, "rb") as f:
        r = requests.post(f"{API}/inspect", files={"file": (p.name, f, "image/jpeg")},
                          data={"product_id": f"TEST-{i:03d}"})
    print(p.name, "->", r.status_code, r.json().get("status"))

# 2. Show what was saved
print("\nSaved inspections:")
for row in requests.get(f"{API}/inspections").json():
    print(row["id"], row["product_id"], row["system_status"], row["final_status"])

# 3. Show the stats
print("\nStats:", requests.get(f"{API}/stats").json())