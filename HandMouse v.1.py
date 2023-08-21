import cv2
import mediapipe as med
import time
import pyautogui 

cam = cv2.VideoCapture(0)
mpHands = med.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = med.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=3)
pTime = 0
cTime = 0

screenWidth, screenHeight = pyautogui.size()

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
                wristX = int(handLms.landmark[0].x * imgWidth)
                wristY = int(handLms.landmark[0].y * imgHeight)
                indexFingerTipX = int(handLms.landmark[8].x * imgWidth)
                indexFingerTipY = int(handLms.landmark[8].y * imgHeight)
                indexFingerX = int(handLms.landmark[5].x * imgWidth)
                indexFingerY = int(handLms.landmark[5].y * imgHeight)
                indexFingerMiddleX = int(handLms.landmark[6].x * imgWidth)
                indexFingerMiddleY = int(handLms.landmark[6].y * imgHeight)

                fingerTipDistance = ((indexFingerTipX - indexFingerX) ** 2 + (indexFingerTipY - indexFingerY) ** 2) ** 0.5
                fingerDistance = ((indexFingerX - indexFingerMiddleX) ** 2 + (indexFingerY - indexFingerMiddleY) ** 2) ** 0.5

                mouseX = screenWidth - (wristX / imgWidth) * screenWidth
                mouseY = (wristY / imgHeight) * screenHeight
                pyautogui.moveTo(mouseX, mouseY)

                if fingerTipDistance < fingerDistance:
                    
                    if not mouseDown:
                        pyautogui.mouseDown()
                        mouseDown = True
                else:
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
