import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load the YOLO models
object_detection_model = YOLO("best.pt")  # Your object detection model
digit_recognition_model = YOLO("best (1).pt")  # Your digit recognition model

# Open the video file
video_path = "MVI_5340.MP4"
cap = cv2.VideoCapture(video_path)

# Video writing setup
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
out = cv2.VideoWriter('MVI_5340_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60.0, size)

# Create 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

def add_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                             font_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    bg_left = position[0] - padding
    bg_top = position[1] - text_h - padding
    bg_right = position[0] + text_w + padding
    bg_bottom = position[1] + padding
    cv2.rectangle(img, (bg_left, bg_top), (bg_right, bg_bottom), bg_color, -1)
    cv2.putText(img, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

def is_valid_wagon_number(number):
    with open('number.txt', 'r') as f:
        valid_numbers = set(f.read().splitlines())
    return number in valid_numbers

# Set to store unique numbers
unique_numbers = set()

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame_count += 1

        results = object_detection_model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_frame = frame[y1:y2, x1:x2]

            digit_results = digit_recognition_model(cropped_frame)
            detected_digits = []
            for r in digit_results:
                for c in r.boxes.cls:
                    detected_digits.append(str(int(c)))

            digit_boxes = digit_results[0].boxes.xyxy.cpu().numpy()
            sorted_indices = digit_boxes[:, 0].argsort()
            sorted_digits = [detected_digits[i] for i in sorted_indices]
            ocr_digits = ''.join(sorted_digits)

            if len(ocr_digits) == 8:
                if ocr_digits not in unique_numbers:
                    unique_numbers.add(ocr_digits)
                    print(f"Found new 8-digit number: {ocr_digits}")
                    
                    # Write to detected_numbers.txt
                    with open('detected_numbers.txt', 'a') as f:
                        f.write(f"{ocr_digits}\n")
                    
                    # Check if it's a valid wagon number
                    if is_valid_wagon_number(ocr_digits):
                        with open('detected_vagon_numbers.txt', 'a') as f:
                            f.write(f"{ocr_digits}\n")
                        
                        # Save the image with the wagon number
                        img_filename = f"images/{ocr_digits}.jpg"
                        cv2.imwrite(img_filename, frame)
                        print(f"Saved image: {img_filename}")

                add_text_with_background(annotated_frame, ocr_digits, (x1, y1 - 10),
                                         font_scale=1, font_color=(255, 255, 255),
                                         bg_color=(0, 128, 0), thickness=2, padding=5)

        out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()