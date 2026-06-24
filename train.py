from ultralytics import YOLO

model = YOLO(r"C:\Users\CHIRNJEEVI\OneDrive\Desktop\Text_detection\icdar2013\yolo11s.pt")

model.train(
    data=r"C:\Users\CHIRNJEEVI\OneDrive\Desktop\Text_detection\icdar2015\icdar2015\data.yaml",
    epochs=10,           # reduce from 50 to 30
    imgsz=640,           # reduce image size from 640 to 416
    batch=16,             # reduce batch size if running on CPU (or increase if on GPU)
    workers=4,           # lower workers for CPU; increase if GPU + enough cores
    device='cpu',        # change to 'cuda' if using GPU
    save=True,+
    name="yolov11_small",  # name of the run
    verbose=False,
    amp=False,           # disable AMP (it's already off on CPU)           
    dropout=0.1          # slight regularization to prevent overfitting in fast runs
)
