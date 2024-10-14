import cv2
from ultralytics import YOLO
import numpy as np

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

def add_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                             font_color=(255, 255, 255), bg_color=(0, 0, 0), thickness=2, padding=5):
    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Calculate background rectangle dimensions
    bg_left = position[0] - padding
    bg_top = position[1] - text_h - padding
    bg_right = position[0] + text_w + padding
    bg_bottom = position[1] + padding
    
    # Draw background rectangle
    cv2.rectangle(img, (bg_left, bg_top), (bg_right, bg_bottom), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

# Set to store unique numbers
unique_numbers = set()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Analyze the frame using the object detection YOLO model
        results = object_detection_model.track(frame, persist=True)
        
        # Annotate
        annotated_frame = results[0].plot()
        
        # Loop through detected objects
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bbox coordinates
            cropped_frame = frame[y1:y2, x1:x2]  # Crop the section inside the box
            
            # Detect digits in the cropped frame using the digit recognition model
            digit_results = digit_recognition_model(cropped_frame)
            
            # Extract detected digits
            detected_digits = []
            for r in digit_results:
                for c in r.boxes.cls:
                    detected_digits.append(str(int(c)))
            
            # Sort digits based on their x-coordinate to get the correct order
            digit_boxes = digit_results[0].boxes.xyxy.cpu().numpy()
            sorted_indices = digit_boxes[:, 0].argsort()
            sorted_digits = [detected_digits[i] for i in sorted_indices]
            
            ocr_digits = ''.join(sorted_digits)
            
            if len(ocr_digits) == 8:  # If we detected an 8-digit number
                if ocr_digits not in unique_numbers:
                    unique_numbers.add(ocr_digits)
                    print(f"Found new 8-digit number: {ocr_digits}")
                    # Write the new number to the text file
                    with open('detected_numbers.txt', 'a') as f:
                        f.write(f"{ocr_digits}\n")
                
                # Show the number on the annotated frame with background
                add_text_with_background(annotated_frame, ocr_digits, (x1, y1 - 10),
                                         font_scale=1, font_color=(255, 255, 255),
                                         bg_color=(0, 128, 0), thickness=2, padding=5)
        
        # Write the annotated frame
        out.write(annotated_frame)
        
        # Display the frame (if desired)
        # cv2.imshow("YOLO Tracking with Digit Detection", annotated_frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Exit loop at the end of the video
        break

# Clean up video writing and windows
cap.release()
out.release()
cv2.destroyAllWindows()