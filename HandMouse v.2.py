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
                # 获取无名指指尖（第15个关键点）、无名指第一节（第14个关键点）和无名指第二节（第13个关键点）的坐标
                indexFingerX = int(handLms.landmark[4].x * imgWidth)
                indexFingerY = int(handLms.landmark[4].y * imgHeight)
                ringFingerTipX = int(handLms.landmark[15].x * imgWidth)
                ringFingerTipY = int(handLms.landmark[15].y * imgHeight)
                # 计算无名指指尖和无名指第一节之间的距离，以及无名指第一节和无名指第二节之间的距离
                fingerDistance = ((indexFingerX - ringFingerTipX) ** 2 + (indexFingerY - ringFingerTipY) ** 2) ** 0.5

                # 根据手腕的坐标，映射到屏幕上的位置，并移动鼠标
                mouseX = screenWidth - (wristX / imgWidth) * screenWidth
                mouseY = (wristY / imgHeight) * screenHeight
                pyautogui.moveTo(mouseX, mouseY)

                # 如果无名指指尖和无名指第一节之间的距离比无名指第一节和无名指第二节之间的距离还短，则握拳
                if fingerDistance < 10:
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
