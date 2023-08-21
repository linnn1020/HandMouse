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
            
                indexFingerX = int(handLms.landmark[4].x * imgWidth)
                indexFingerY = int(handLms.landmark[4].y * imgHeight)
                ringFingerTipX = int(handLms.landmark[15].x * imgWidth)
                ringFingerTipY = int(handLms.landmark[15].y * imgHeight)
                fingerDistance = ((indexFingerX - ringFingerTipX) ** 2 + (indexFingerY - ringFingerTipY) ** 2) ** 0.5

                mouseX = screenWidth - (wristX / imgWidth) * screenWidth
                mouseY = (wristY / imgHeight) * screenHeight
                pyautogui.moveTo(mouseX, mouseY)

                if fingerDistance < 10:
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
