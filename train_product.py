import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_to_yolo(xml_path, classes, w, h):
    lines = []
    for o in ET.parse(xml_path).getroot().iter("object"):
        name = o.findtext("name").strip()
        if name not in classes:
            raise SystemExit(f"Unknown class '{name}' in {xml_path}")
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        lines.append(f"{classes.index(name)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                     f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines


def build_dataset(src, out, seed):
    src, out = Path(src), Path(out)
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
    labels = {p.stem: p for p in src.rglob("*")
              if p.suffix.lower() in {".xml", ".txt"} and p.stem in images and p.name != "classes.txt"}
    stems = sorted(set(images) & set(labels))
    if not stems:
        raise SystemExit("No image and label pairs found. Check the file names match.")

    xmls = [labels[s] for s in stems if labels[s].suffix == ".xml"]
    if xmls:
        classes = sorted({o.findtext("name").strip()
                          for x in xmls for o in ET.parse(x).getroot().iter("object")})
    else:
        classes = [c.strip() for c in (src / "classes.txt").read_text().splitlines() if c.strip()]
    if not classes:
        raise SystemExit("No defect classes found. Add at least one labelled defect.")

    random.Random(seed).shuffle(stems)
    n = len(stems)
    splits = {"train": stems[:int(.7 * n)],
              "val": stems[int(.7 * n):int(.85 * n)],
              "test": stems[int(.85 * n):]}
    counts = {}
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        ok = 0
        for s in items:
            img = cv2.imread(str(images[s]))
            if img is None:
                print("Skipping unreadable image:", images[s].name)
                continue
            h, w = img.shape[:2]
            if labels[s].suffix == ".xml":
                lines = voc_to_yolo(labels[s], classes, w, h)
            else:
                lines = labels[s].read_text().splitlines()
            shutil.copy(images[s], out / "images" / split / images[s].name)
            (out / "labels" / split / f"{s}.txt").write_text("\n".join(lines))
            ok += 1
        counts[split] = ok

    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(classes)))
    return yaml, counts, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with images and labels")
    ap.add_argument("--name", required=True, help="model name, e.g. mug_v1")
    ap.add_argument("--base", default="yolov8s.pt",
                    help="starting weights; use models\\best.pt to fine-tune the existing model")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu for no GPU")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run_dir = Path("training").resolve()
    yaml, counts, classes = build_dataset(a.src, run_dir / a.name / "dataset", a.seed)
    print("Classes:", classes)
    print("Split sizes (train/val/test):", counts)

    YOLO(a.base).train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                       device=a.device, project=str(run_dir), name=a.name,
                       plots=True, exist_ok=True)

    best = run_dir / a.name / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(yaml), split="test", plots=True,
                            project=str(run_dir), name=f"{a.name}_test", exist_ok=True)
    print(f"\nTEST  precision {m.box.mp:.3f}  recall {m.box.mr:.3f}  "
          f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}")

    Path("models").mkdir(exist_ok=True)
    dest = Path("models") / f"{a.name}.pt"
    shutil.copy(best, dest)
    print(f"Saved {dest}. Test on your real photos before using it.")


if __name__ == "__main__":
    main()
