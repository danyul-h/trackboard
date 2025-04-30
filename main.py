
import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils #render hand landmarks
mp_hands = mp.solutions.hands

#use first device as a video capture
cap = cv2.VideoCapture(0)

#set boundaries
cap.set(3,640)
cap.set(4,640)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        #capture frame by frame in BGR color
        re, frame = cap.read()

        #convert to RGB for mediapipe
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #flip the frame
        frame = cv2.flip(frame, 1)

        #mp processes frame
        results = hands.process(frame)

        # frame.flags.writeable = False

        #draw landmarks on frame
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        #revert back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #display results
        cv2.imshow("image", frame)

        #wait 10ms for a key press, if its q then exit
        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()