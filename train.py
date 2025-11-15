from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")  # bisa ganti yolov8s, yolov8m sesuai GPU / kebutuhan

# Start training
model.train(
    data="data.yaml",  # file YAML tadi
    epochs=50,                   # ganti sesuai keinginan
    imgsz=640,                   # ukuran input gambar
    batch=8,                     # batch size
    name="pakcoy_model"          # nama folder hasil training
)
