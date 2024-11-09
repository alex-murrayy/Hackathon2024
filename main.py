import cv2 
import attacker
import defender
import ball 

cap = cv2.VideoCapture("/Users/alex/Code/Hackathon2024/RPReplay_Final1731175138.mov")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()