from ultralytics import YOLO

# Load model hasil training kamu
model = YOLO("runs/train/pakcoy_model/weights/best.pt")

# Langsung nyalain webcam
model.predict(
    source=0,     # kamera default
    show=True,    # tampilin window
    conf=0.5      # confidence bisa kamu kecilin/besarin
)
