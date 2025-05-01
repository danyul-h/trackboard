
import mediapipe as mp
import numpy as np
import cv2
import tkinter
import math
root = tkinter.Tk() 

from enum import IntEnum
class Hand(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2 
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP =5 
    INDEX_FINGER_PIP =6 
    INDEX_FINGER_DIP =7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13  
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP =16 
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP  = 20

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

def extend_state(hand):
    finger_tips = [int(Hand.INDEX_FINGER_TIP), int(Hand.MIDDLE_FINGER_TIP), int(Hand.RING_FINGER_TIP), int(Hand.PINKY_TIP)]
    finger_states = [0, 0, 0, 0, 0]
    if hand.landmark[int(Hand.THUMB_TIP)].x > hand.landmark[int(Hand.THUMB_TIP)-1].x:
        finger_states[0] = 1
    for i, tip in enumerate(finger_tips):
        if hand.landmark[tip].y < hand.landmark[tip-2].y:
            finger_states[i +1 ] = 1
    return finger_states

def check_extend(states, checks):
    for i, state in enumerate(states):
        if state != checks[i]: return False
    return True


mp_drawing = mp.solutions.drawing_utils #render hand landmarks
mp_hands = mp.solutions.hands

#use first device as a video capture
cap = cv2.VideoCapture(0)

#set boundaries
cap.set(3,640)
cap.set(4,640)

screen_width = root.winfo_screenwidth()  
screen_height = root.winfo_screenheight()
newy = screen_height
newx = screen_width

scale = 0.5

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

        h,w,c = frame.shape

        #draw landmarks on frame
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                extend_states = extend_state(hand)

                thumblm = hand.landmark[int(Hand.THUMB_TIP)]
                thumbx, thumby= int(thumblm.x*w), int(thumblm.y*h)

                indexlm = hand.landmark[int(Hand.INDEX_FINGER_TIP)]
                indexx, indexy = int(indexlm.x*w), int(indexlm.y*h)

                wristlm = hand.landmark[9]

                print(extend_states)
                if check_extend(extend_states, [1,1,1,1,1]):
                    newx, newy = wristlm.x*screen_width, wristlm.y*screen_height

                if check_extend(extend_states, [1,1,0,0,0]):
                    cv2.line(frame, (thumbx, thumby), (indexx, indexy), (0,0,255), 9)
                    scale = math.sqrt(math.pow(thumbx-indexx, 2) + math.pow(thumby-indexy, 2)) / 300
                    

        #revert back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame2 = rescale_frame(frame, scale)
        #display results
        cv2.imshow("image", frame2)

        # Move the window to the specified position
   
        cv2.moveWindow("image", int(newx), int(newy))  # Move to x=100, y=200

        #wait 10ms for a key press, if its q then exit
        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()