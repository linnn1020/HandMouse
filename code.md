import cv2
import mediapipe as mp
import math
import time  
import pyautogui as gui
import multiprocessing as mup
        
pTime =0
fps =0
screenWidth, screenHeight = gui.size()
mouseX=0
mouseY=0

def vector_2d_angle(v1,v2):
    
        #求解二维向量角度
    
    v1_x=v1[0]
    v1_y=v1[1]
    v2_x=v2[0]
    v2_y=v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ =65535.
    if angle_ > 180.:
        angle_ = 65535.
    return angle_

def hand_angle(hand_):
    
        #獲取對應手相關向量的二維角度,根據角度確定手勢
    
    angle_list = []
    #---------------------------- thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    #---------------------------- index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    #---------------------------- middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    #---------------------------- ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    #---------------------------- pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

def h_gesture(angle_list):

        # 二維約束定義手勢

    thr_angle = 65.
    thr_angle_thumb = 53.
    thr_angle_s = 49.
    gesture_str = None
    if 65535. not in angle_list:
        if (angle_list[0]>thr_angle_thumb) and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "fist"
        elif (angle_list[0]<thr_angle_s) and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]<thr_angle_s):
            gesture_str = "five"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "gun"
        elif (angle_list[0]>5)  and (angle_list[1]<thr_angle_s) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "one"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]<thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]<thr_angle_s) and (angle_list[4]>thr_angle):
            gesture_str = "three"
        elif (angle_list[0]<thr_angle_s)  and (angle_list[1]>thr_angle) and (angle_list[2]>thr_angle) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "thumbUp"
        elif (angle_list[0]>thr_angle_thumb)  and (angle_list[1]<thr_angle_s) and (angle_list[2]<thr_angle_s) and (angle_list[3]>thr_angle) and (angle_list[4]>thr_angle):
            gesture_str = "two"
    return gesture_str

def detect():
    global mouseX
    global mouseY
    global screenHeight
    global screenWidth

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    cv2.namedWindow('Hands', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hands', int(screenWidth), int(screenHeight))  # 设置窗口大小
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    hand_local.append((x,y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    if gesture_str=="two":
                        slope =index_anglecounter(hand_landmarks.landmark[6].x,hand_landmarks.landmark[6].y,hand_landmarks.landmark[8].x,hand_landmarks.landmark[8].y)
                        if slope <=2.5 and slope >=0:
                            gesture_str="left"
                        elif slope >= -5 and slope <= -1:
                            gesture_str="right"
                    cv2.putText(frame,gesture_str,(0,100),0,1.3,(0,0,255),3)

                    xpoint=(hand_landmarks.landmark[0].x +hand_landmarks.landmark[5].x+hand_landmarks.landmark[17].x)/3
                    ypoint=(hand_landmarks.landmark[0].y +hand_landmarks.landmark[5].y+hand_landmarks.landmark[17].y)/3
                    mouseX = (xpoint * (screenWidth) / (screenWidth)) * screenWidth
                    mouseY = (ypoint * (screenHeight) / (screenHeight)) * screenHeight
                    
                    mouseCrontrol(gesture_str)

        fps_count()
        cv2.putText(frame, f"FPS : {int(fps)}", (0, 50),0, 1, (255, 225, 225), 2)
        cv2.imshow('Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()

def index_anglecounter(x1,y1,x2,y2):
    angle=math.atan2(y2-y1,x2-x1)
    slope = math.tan(angle)
    return slope

def fps_count():
    global pTime 
    global fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

def mouseCrontrol(mode):
    global mouseY
    global mouseX
    match mode:
        case "five":
            gui.moveTo(mouseX,mouseY)
            gui.mouseUp()
            
        case "right":
            gui.scroll(25)
        case "left":
            gui.scroll(-25)
        case "fist":
            gui.moveTo(mouseX,mouseY)
            gui.mouseDown()

if __name__ == '__main__':
    detect()    
