import random
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path("data/test_images")
dst.mkdir(parents=True, exist_ok=True)

by_class = {}
for p in src.rglob("*"):
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        by_class.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)

if not by_class:
    raise SystemExit(f"ERROR: no images found under {src}")

for cls, files in by_class.items():
    for p in random.sample(files, min(2, len(files))):
        shutil.copy(p, dst / p.name)

print("Classes found:", list(by_class))
print("Copied to", dst)