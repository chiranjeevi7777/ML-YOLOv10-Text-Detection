from ultralytics import YOLO
import easyocr
import cv2

# Load YOLO text detector
model = YOLO("best.pt")

# OCR Reader
reader = easyocr.Reader(['en'])

image_path = "test.jpg"
image = cv2.imread(image_path)

results = model.predict(image)

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        roi = image[y1:y2, x1:x2]

        text = reader.readtext(roi, detail=0)

        print("Detected Text:", " ".join(text))

        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

cv2.imshow("Detected Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
