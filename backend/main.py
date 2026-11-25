import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import db
from product_inspection import inspect_for_product

ROOT = Path(__file__).resolve().parents[1]
UPLOADS = ROOT / "data" / "uploads"
ANNOTATED = ROOT / "reports" / "annotated"
UPLOADS.mkdir(parents=True, exist_ok=True)
ANNOTATED.mkdir(parents=True, exist_ok=True)

db.init_db()
app = FastAPI(title="AI Defect Inspection API", version="1.0.0")
app.mount("/files", StaticFiles(directory=str(ANNOTATED)), name="files")


class Review(BaseModel):
    decision: str
    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=1000)


@app.get("/")
def root():
    return {"name": "AI Defect Inspection API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products")
def products():
    from products import load_products
    return load_products()


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), product_id: str = Form("")):
    allowed = {"image/jpeg", "image/png", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(415, "Upload a JPG, PNG or BMP image")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded image is empty")
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds the 15 MB upload limit")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The uploaded file is not a valid image")

    try:
        status, detections, annotated, used_product = inspect_for_product(image, product_id)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, "Trained model is missing. Train the model first; see README.md.") from exc

    filename = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.jpg"
    original_path = UPLOADS / filename
    annotated_path = ANNOTATED / filename
    if not cv2.imwrite(str(original_path), image) or not cv2.imwrite(str(annotated_path), annotated):
        raise HTTPException(500, "Could not save inspection images")

    inspection_id = db.add_inspection(used_product, filename, status, detections)
    return {
        "id": inspection_id,
        "product": used_product,
        "status": status,
        "detections": detections,
        "image_file": filename,
        "annotated_image": f"/files/{filename}",
    }


@app.get("/inspections")
def list_inspections(limit: int = 50, status: str | None = None):
    if not 1 <= limit <= 500:
        raise HTTPException(400, "limit must be between 1 and 500")
    return db.list_inspections(limit, status)


@app.get("/inspections/{inspection_id}")
def get_inspection(inspection_id: int):
    row = db.get_inspection(inspection_id)
    if not row:
        raise HTTPException(404, "Inspection not found")
    return row


@app.post("/inspections/{inspection_id}/review")
def review(inspection_id: int, body: Review):
    if body.decision not in ("Pass", "Fail"):
        raise HTTPException(400, "decision must be 'Pass' or 'Fail'")
    if not db.add_review(inspection_id, body.decision, body.reviewer.strip(), body.note.strip()):
        raise HTTPException(404, "Inspection not found")
    return db.get_inspection(inspection_id)


@app.get("/stats")
def stats():
    return db.stats()
