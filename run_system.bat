@echo off
set DEFECT_MODEL=models\best.pt
set DEFECT_API=http://127.0.0.1:8000
start "Defect backend" cmd /k "call venv\Scripts\activate && uvicorn backend.main:app --host 127.0.0.1 --port 8000"
start "Defect interface" cmd /k "call venv\Scripts\activate && streamlit run app\streamlit_app.py --server.port 8501"
echo Interface: http://localhost:8501
