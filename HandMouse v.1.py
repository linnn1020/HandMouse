import cv2
import mediapipe as med
import time
import pyautogui # 导入pyautogui库

cam = cv2.VideoCapture(0)
mpHands = med.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = med.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=3)
pTime = 0
cTime = 0

# 获取屏幕的宽度和高度
screenWidth, screenHeight = pyautogui.size()

# 设置一个变量，用来记录是否已经按下鼠标左键
mouseDown = False

while True:
    ret, img = cam.read()
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                # 获取手腕的坐标（第0个关键点）
                wristX = int(handLms.landmark[0].x * imgWidth)
                wristY = int(handLms.landmark[0].y * imgHeight)
                # 获取食指指尖（第8个关键点）、食指第一节（第5个关键点）和食指第二节（第6个关键点）的坐标
                indexFingerTipX = int(handLms.landmark[8].x * imgWidth)
                indexFingerTipY = int(handLms.landmark[8].y * imgHeight)
                indexFingerX = int(handLms.landmark[5].x * imgWidth)
                indexFingerY = int(handLms.landmark[5].y * imgHeight)
                indexFingerMiddleX = int(handLms.landmark[6].x * imgWidth)
                indexFingerMiddleY = int(handLms.landmark[6].y * imgHeight)

                # 计算食指指尖和食指第一节之间的距离，以及食指第一节和食指第二节之间的距离
                fingerTipDistance = ((indexFingerTipX - indexFingerX) ** 2 + (indexFingerTipY - indexFingerY) ** 2) ** 0.5
                fingerDistance = ((indexFingerX - indexFingerMiddleX) ** 2 + (indexFingerY - indexFingerMiddleY) ** 2) ** 0.5

                # 根据手腕的坐标，映射到屏幕上的位置，并移动鼠标
                mouseX = screenWidth - (wristX / imgWidth) * screenWidth
                mouseY = (wristY / imgHeight) * screenHeight
                pyautogui.moveTo(mouseX, mouseY)

                # 如果食指指尖和食指第一节之间的距离比食指第一节和食指第二节之间的距离还短，则握拳
                if fingerTipDistance < fingerDistance:
                    # 如果还没有按下鼠标左键，则按下鼠标左键，并设置mouseDown为True
                    if not mouseDown:
                        pyautogui.mouseDown()
                        mouseDown = True
                else:
                    # 如果已经按下鼠标左键，则松开鼠标左键，并设置mouseDown为False
                    if mouseDown:
                        pyautogui.mouseUp()
                        mouseDown = False

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
