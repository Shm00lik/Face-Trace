import cv2
from models import Rectangle, Point, PointKalmanFilter, RectangleKalmanFilter

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

K = 20

center_kalman_filter = PointKalmanFilter(K)
rect_kalman_filter = RectangleKalmanFilter(K)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break  # Exit if no frame is captured

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    faces_list = sorted(
        [Rectangle(x, y, w, h) for x, y, w, h in faces],
        key=lambda f: f.get_area(),
        reverse=True,
    )

    if len(faces_list) > 0:
        rect_kalman_filter.update(faces_list[0])
        center_kalman_filter.update(faces_list[0].get_center())

    face = rect_kalman_filter.get()

    corners = face.get_corners()

    cv2.rectangle(
        frame,
        corners.top_left.as_tuple(),
        corners.bottom_right.as_tuple(),
        (0, 255, 0),
        4,
    )

    center = center_kalman_filter.get()

    cv2.circle(frame, (int(center.x), int(center.y)), 5, (0, 0, 255), 10)

    OFFSET = 50  # Pixels
    RES = 16 / 9

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_y_cropped_frame = max(corners.top_left.y - OFFSET, 0)
    bottom_y_cropped_frame = min(corners.bottom_right.y + OFFSET, frame_height)
    cropped_height = bottom_y_cropped_frame - top_y_cropped_frame

    left_x_cropped_frame = max(corners.top_left.x - OFFSET, 0)
    right_x_cropped_frame = min(
        left_x_cropped_frame + int(cropped_height * RES),
        frame_width,
    )

    print(
        f"ty: {top_y_cropped_frame}, by: {bottom_y_cropped_frame}, H: {cropped_height}, hr: {cropped_height * RES}, lx: {left_x_cropped_frame}, rx: {right_x_cropped_frame}"
    )

    cropped_frame = frame[
        top_y_cropped_frame:bottom_y_cropped_frame,
        left_x_cropped_frame:right_x_cropped_frame,
    ]

    cv2.imshow("Webcam Feed", frame)  # Show the frame
    cv2.imshow("Cropped Feed", cropped_frame)  # Show the frame

    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
