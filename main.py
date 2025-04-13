import cv2
from models import Rectangle, Point, PointKalmanFilter

### HOW TO STREAM THE FEED TO ZOOM:
### 1. Run this program.
### 2. Open OBS.
### 3. Start Virtual Camera on OBS.
### 4. Use the OBS Virtuca

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
)

kalman_filter = PointKalmanFilter(35)


while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break  # Exit if no frame is captured

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        rect = Rectangle(x, y, w, h)

        kalman_filter.update(rect.get_center())

        center = kalman_filter.get()

        cv2.circle(frame, (int(center.x), int(center.y)), 5, (0, 0, 255), 10)

    cv2.imshow("Webcam Feed", frame)  # Show the frame

    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
