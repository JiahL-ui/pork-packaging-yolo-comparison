from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.yaml")
    model.train(
        data="data/dataset.yaml",
        epochs=30,
        imgsz=416,
        batch=16,
        optimizer="AdamW",
        mosaic=1.0
    )
