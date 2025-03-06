import cv2

# Open webcam (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break  # Exit if no frame is captured

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=15, minSize=(40, 40)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow("Webcam Feed", frame)  # Show the frame

    # Press 'q' to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
