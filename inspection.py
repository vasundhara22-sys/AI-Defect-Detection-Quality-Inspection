from ultralytics import YOLO

import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = YOLO(config.MODEL_PATH)
    return _model


def decide(confidences):
    """Apply the configurable Pass / Fail / Review Required rules."""
    if not confidences:
        return "Pass"
    top = max(confidences)
    if top >= config.FAIL_CONF:
        return "Fail"
    if top >= config.REVIEW_CONF:
        return "Review Required"
    return "Pass"


def inspect_image(image):
    """image: a BGR numpy array. Returns (status, detections, annotated_image)."""
    model = get_model()
    result = model.predict(image, conf=config.REVIEW_CONF, verbose=False)[0]

    detections = []
    for b in result.boxes:
        detections.append({
            "class": model.names[int(b.cls)],
            "confidence": round(float(b.conf), 3),
            "box": [round(v, 1) for v in b.xyxy[0].tolist()],
        })

    status = decide([d["confidence"] for d in detections])
    return status, detections, result.plot(font_size=8, line_width=1)