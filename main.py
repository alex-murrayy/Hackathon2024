import cv2

cap = cv2.VideoCapture()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    pupil_detector = pupil.PupilDetection(frame)
    pupil_detector.start_detection()
    

    if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()