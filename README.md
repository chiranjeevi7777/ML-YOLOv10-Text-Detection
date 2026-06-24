# YOLOv10 Scene Text Detection 🔍
A Deep Learning pipeline utilizing the **YOLOv10** architecture for scene text detection and bounding box localization, designed for document processing, OCR systems, and visual data extraction.

## 🚀 Features
- **Modern Object Detection:** Leverages **YOLOv10** (specifically YOLOv10n) for fast, highly accurate, and NMS-free object detection.
- **Custom-Labeled Dataset:** Trained on custom scene text images labeled using `LabelImg` (annotated in YOLO format).
- **Comprehensive Training and Inference Pipelines:**
  - `model.py` - Contains the training pipeline configuration, data loader configuration, and training execution loop.
  - `Pred.py` - Implements the inference logic to read images, run text detection, and visualize localized bounding boxes.

## 🛠️ Tech Stack & Libraries
- **Architecture:** YOLOv10 (YOLOv10n)
- **Frameworks:** PyTorch, Ultralytics
- **Image Processing:** OpenCV, PIL
- **Annotation Tool:** LabelImg

## 📁 Repository Structure
- `model.py` — Deep learning model training & hyperparameters configuration script.
- `Pred.py` — Bounding box prediction, text detection, and inference script.
- `README.md` — Project overview and setup instructions.

## 📦 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.9+ and PyTorch installed. Install the required packages:
```bash
pip install ultralytics opencv-python pillow
```

### 2. Dataset Setup
Organize your dataset in the standard YOLO format:
```
dataset/
  ├── images/
  │    ├── train/
  │    └── val/
  └── labels/
       ├── train/
       └── val/
```

### 3. Model Training
To train the YOLOv10n model on your custom dataset:
```bash
python model.py
```

### 4. Running Text Localization Inference
To run predictions and output text bounding boxes on an image:
```bash
python Pred.py --image path/to/your/image.jpg
```
