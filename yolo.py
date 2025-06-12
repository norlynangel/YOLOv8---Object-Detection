from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("best.pt")

# Open webcam feed (0 for default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to process the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run inference on the webcam frame
    results = model.predict(source=frame, save=False, show=False)

    # Draw bounding boxes and labels on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        conf = box.conf[0]  # Get confidence score
        cls = box.cls[0]  # Get class label
        label = f"{results[0].names[int(cls)]} {conf:.2f}"  # Class name and confidence

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Webcam Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
