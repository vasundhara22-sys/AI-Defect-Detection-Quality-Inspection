import torch, cv2, ultralytics, fastapi, streamlit
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("GPU available:", torch.cuda.is_available())