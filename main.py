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

K = 25

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

    face = rect_kalman_filter.get()

    corners = face.get_corners()
    center = face.get_center()

    cv2.rectangle(
        frame,
        corners.top_left.as_tuple(),
        corners.bottom_right.as_tuple(),
        (0, 255, 0),
        4,
    )

    cv2.circle(frame, center.as_tuple(), 5, (0, 0, 255), 10)

    OFFSET = 50  # Pixels
    RES = 16 / 9

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_y_cropped_frame = max(corners.top_left.y - OFFSET, 0)
    bottom_y_cropped_frame = min(corners.bottom_right.y + OFFSET, frame_height)

    cropped_height = bottom_y_cropped_frame - top_y_cropped_frame
    cropped_width = int(cropped_height * RES)

    exceeds_left = int(center.x - cropped_width / 2) < 0
    exceeds_right = int(center.x + cropped_width / 2) > frame_width

    left_x_cropped_frame = 0
    right_x_cropped_frame = frame_width

    if exceeds_left:
        right_x_cropped_frame = min(cropped_width, frame_width)
    elif exceeds_right:
        left_x_cropped_frame = max(frame_width - cropped_width, 0)
    else:
        left_x_cropped_frame = int(center.x - cropped_width / 2)
        right_x_cropped_frame = int(center.x + cropped_width / 2)

    cv2.line(
        frame,
        (center.x, top_y_cropped_frame),
        (center.x, bottom_y_cropped_frame),
        (255, 255, 0),
        4,
    )

    cv2.line(
        frame,
        (int(center.x - cropped_width / 2), center.y),
        (int(center.x + cropped_width / 2), center.y),
        (255, 255, 0),
        4,
    )

    cropped_frame = frame[
        top_y_cropped_frame:bottom_y_cropped_frame,
        left_x_cropped_frame:right_x_cropped_frame,
    ]

    resized = cv2.resize(cropped_frame, (1920, 1080))
    cv2.imshow("Webcam Feed", frame)  # Show the frame
    cv2.imshow("Cropped Feed", resized)  # Show the frame

    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
