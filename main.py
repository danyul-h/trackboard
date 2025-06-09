
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

projection_matrix = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,0]
])

tetrahedron = np.array([
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
tetrahedron_connections = np.array([
    [0, 1],
    [0, 2],
    [1, 2],
    [0, 3],
    [1, 3],
    [2, 3]
])

cube = np.array([
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
    [-1,-1,-1],
    [1,-1,-1],
    [1,1,-1],
    [-1,1,-1]
])


def connect_points(i, j, points, frame):
    cv2.line(frame, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), (255,0,0), 2)

mp_drawing = mp.solutions.drawing_utils #render hand landmarks
mp_hands = mp.solutions.hands

#use first device as a video capture
cap = cv2.VideoCapture(0)

#set boundaries
cap.set(3,640)
cap.set(4,640)
circle_pos = (320, 240)
screen_width = root.winfo_screenwidth()  
screen_height = root.winfo_screenheight()


dots = []

scale = 100

register = False

angle = 0

shape = cube
projected_points = [
    [n, n] for n in range(len(shape))
]

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

        rotation_z = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])

        rotation_y = np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)],
        ])
        rotation_x = np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
        ])

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
                if check_extend(extend_states, [1,1,1,1,1]): angle+=0.05
                if check_extend(extend_states, [0,0,0,0,0]): angle-=0.05
                elif check_extend(extend_states, [1,1,0,0,0]):
                    cv2.line(frame, (thumbx, thumby), (indexx, indexy), (255,0,0), 2)
                    scale = math.sqrt(math.pow(thumbx-indexx, 2) + math.pow(thumby-indexy, 2))
                    angle+=0.05
        
        i=0
        for point in shape:
            rotated2d = np.dot(rotation_z, point.reshape((3,1)))
            rotated2d = np.dot(rotation_y, rotated2d)
            rotated2d = np.dot(rotation_x, rotated2d)

            projected2d = np.dot(projection_matrix, rotated2d)

            x = int(projected2d[0][0] * scale) + circle_pos[0]
            y = int(projected2d[1][0] * scale) + circle_pos[1]
            dot = (int(x),int(y))
            projected_points[i] = [x,y]
            cv2.circle(frame, dot, 2, (255,0,0), 2)
            i+=1
        if shape.all() == tetrahedron.all():
            for i,j in tetrahedron_connections:
                connect_points(i, j, projected_points, frame)
        elif shape.all() == cube.all():
            for p in range(4):
                connect_points(p, (p+1) % 4, projected_points, frame)
                connect_points(p+4, ((p+1) % 4) + 4, projected_points, frame)
                connect_points(p, (p+4), projected_points, frame)

        #revert back to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #display resultsq
        cv2.imshow("image", frame)
        #wait 10ms for a key press, if its q then exit
        if cv2.waitKey(10) == ord("1"):
            shape = cube
            projected_points = [
                [n, n] for n in range(len(cube))
            ]
        if cv2.waitKey(10) == ord("2"):
            shape = tetrahedron
            projected_points = [
                [n, n] for n in range(len(tetrahedron))
            ]
        if cv2.waitKey(10) == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()