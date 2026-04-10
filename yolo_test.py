from ultralytics import YOLO
import cv2
import os

# Linux paths (WSL)
model_path = "/home/kiit/crowd_detection/YOLO/DEMO/models/weights.pt"
input_folder = "/home/kiit/crowd_detection/YOLO/DEMO/test_img/yolo_test"

# Load model
model = YOLO(model_path)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for filename in os.listdir(input_folder):

    # skip already processed images
    if filename.endswith("_op.jpg"):
        continue

    if filename.lower().endswith(valid_ext):

        input_path = os.path.join(input_folder, filename)

        image = cv2.imread(input_path)

        # detection
        results = model(
            image,
            conf=0.15,
            iou=0.4,
            max_det=3000,
            imgsz=1280
        )

        boxes = results[0].boxes
        count = len(boxes)

        # draw boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # add count
        cv2.putText(
            image,
            f"Heads = {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # save output
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(input_folder, f"{name}_op.jpg")

        cv2.imwrite(output_path, image)

        print(f"{filename} → Heads detected: {count}")

print("\nAll images processed.")