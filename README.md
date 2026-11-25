# AI-Powered Automated Defect Detection and Quality Inspection System

## Project status

This repository combines the supplied project files into one runnable application structure. It includes:
- YOLOv8 training pipeline for NEU-DET steel-surface defects
- FastAPI inspection API
- Streamlit interface for image, camera and video inspection
- SQLite inspection history, dashboard and human review
- sample images, demo video and robustness/hard-case image sets

**Important:** the uploaded archives did not contain a trained defect model (`models/neu_surface.pt`). `yolov8s.pt` is a general pretrained starting checkpoint, not a trained steel-defect detector. Train the project model before running inspection. The system is a prototype, not production-ready.

## Defect classes

`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`.

## Environment

Use Python 3.11 or 3.12. Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build suitable for your machine from the official PyTorch install selector if the default dependency resolution does not provide the correct CPU/CUDA build.

## Train the steel-surface model

The supplied `data/raw` folder contains the NEU-DET image/annotation pairs. `train_product.py` converts Pascal VOC XML annotations to YOLO labels, creates train/validation/test splits, trains YOLOv8s, evaluates on the held-out test split and copies the best weights to `models/neu_surface.pt`.

```powershell
python train_product.py data/raw --name neu_surface --base yolov8s.pt --epochs 50 --imgsz 640 --batch 8 --device cpu
```

For a CUDA GPU, use `--device 0` and adjust batch size to available VRAM. Training can take a long time on CPU. Keep the test split separate from training and do not tune thresholds on the test set.

## Run the application

Open two terminals in the project root, with `.venv` activated in both.

Terminal 1:

```powershell
python -m uvicorn backend.main:app --reload
```

Terminal 2:

```powershell
python -m streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. API documentation is at `http://127.0.0.1:8000/docs`.

## Main pages

- Inspect: upload a JPG/PNG/BMP image or capture from a supported camera
- Video: sample frames from a conveyor video and inspect them
- Review: confirm or correct uncertain system decisions
- Dashboard: inspection totals, defect counts and failure trends
- History: browse and export saved inspection results

## Decision rules

For the highest detected defect confidence:
- below `REVIEW_CONF` or no detections: Pass
- `REVIEW_CONF` through below `FAIL_CONF`: Review Required
- at or above `FAIL_CONF`: Fail

Defaults: `REVIEW_CONF=0.15`, `FAIL_CONF=0.60`. These are provisional operating thresholds, not universal quality standards. Adjust only using a representative validation set containing both good and defective products.

## Tests

```powershell
python -m pytest tests -v
```

## Data and evaluation discipline

The supplied README reported a prior NEU-DET evaluation on 270 test images and 630 labelled defects: precision 0.700, recall 0.703, mAP50 0.753 and mAP50-95 0.435. Those figures are historical claims from the supplied project documentation; they have not been independently reproduced in this merged package. Re-run training and evaluation and preserve the generated Ultralytics metrics before reporting them.

The supplied data includes NEU-DET examples, small drawing-based examples, cup images, hard cases and robustness variants. These additional sets are not interchangeable with the NEU-DET test split and should not be represented as a representative factory test set without appropriate labels and validation.

## Known limitations and responsible use

- The NEU-DET dataset is steel-surface imagery; performance on other product types is unknown.
- The supplied dataset has no reliable defect-free production distribution for measuring factory false-alarm rates.
- Crazing was the weakest class in the historical report.
- A confidence threshold does not guarantee a calibrated probability of defect.
- Conveyor frame sampling is a demo workflow, not a factory-grade trigger/tracking system.
- Authentication, role-based access, audit controls and secure deployment need to be added before multi-user use.
- Keep human review for uncertain and high-impact decisions. Validate on representative images from the actual product, camera, lighting and production line before operational use.
