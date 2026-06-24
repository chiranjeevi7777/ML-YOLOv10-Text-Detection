from ultralytics import YOLO


def train_model():
    # Load pretrained YOLO model
    model = YOLO("yolo11s.pt")

    # Train model
    model.train(
        data="data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,
        workers=4,
        device="cpu",      # change to "cuda" if GPU available
        save=True,
        name="yolov11_text_detection",
        verbose=False,
        amp=False,
        dropout=0.1
    )

    # Save best model
    print("Training completed.")


if __name__ == "__main__":
    train_model()
