from functools import lru_cache
from pathlib import Path

from products import get_product


@lru_cache(maxsize=4)
def load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    from ultralytics import YOLO
    return YOLO(str(path))


def decide(top, fail_conf, review_conf):
    if top is None:
        return "Pass"
    if top >= fail_conf:
        return "Fail"
    if top >= review_conf:
        return "Review Required"
    return "Pass"


def inspect_for_product(image, product_id=""):
    product_id = product_id or "steel_surface"
    product = get_product(product_id)
    model = load_model(product["model"])
    result = model.predict(image, conf=product["review_conf"], verbose=False)[0]

    detections = []
    for box in result.boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        })

    top = max((d["confidence"] for d in detections), default=None)
    status = decide(top, product["fail_conf"], product["review_conf"])
    return status, detections, result.plot(font_size=8, line_width=1), product_id
