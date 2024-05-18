from ultralytics import YOLO

# Load the pretrained model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data="grayscale_config.yaml", epochs=100)
