import os

MODEL_PATH = os.environ.get("DEFECT_MODEL", "models/neu_surface.pt")
FAIL_CONF = float(os.environ.get("FAIL_CONF", "0.60"))
REVIEW_CONF = float(os.environ.get("REVIEW_CONF", "0.15"))

if not 0.0 <= REVIEW_CONF <= FAIL_CONF <= 1.0:
    raise ValueError("Thresholds must satisfy 0 <= REVIEW_CONF <= FAIL_CONF <= 1")
