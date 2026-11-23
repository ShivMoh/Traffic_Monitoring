from ultralytics import YOLO # Load an official or custom model

model = YOLO("yolov8n.pt")  # Load an official Detect model
# Perform tracking with the model


results = model.track("https://www.youtube.com/watch?v=7XH8-0K1qpA", show=True)  


