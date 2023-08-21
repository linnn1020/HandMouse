import tensorflow as tf
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image 
import pyautogui 
import mediapipe as med



mpHands = med.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = med.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=3)
fontpath = 'NotoSansTC-Regular.ttf'         
mouseDown = False
screenWidth, screenHeight = pyautogui.size()
model = tf.keras.models.load_model('keras_model.h5', compile=False)  #Load model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)          

def text(text):   
    global img   
    font = ImageFont.truetype(fontpath, 50) 
    imgPil = Image.fromarray(img)            
    draw = ImageDraw.Draw(imgPil)          
    draw.text((0, 0), text, fill=(255, 255, 255), font=font) 
    img = np.array(imgPil)                   

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgHeight = frame.shape[0]
    imgWidth = frame.shape[1]
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                # 获取手腕的坐标（第0个关键点）
            wristX = int(handLms.landmark[0].x * imgWidth)
            wristY = int(handLms.landmark[0].y * imgHeight)
            mouseX = screenWidth - (wristX / imgWidth) * screenWidth
            mouseY = (wristY / imgHeight) * screenHeight
    img = cv2.resize(frame , (398, 224))
    img = img[0:224, 80:304]
    image_array = np.asarray(img)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    a,b,none= prediction[0]
    if a>0.92:
        text('Move')  
        pyautogui.mouseUp()
        pyautogui.moveTo(mouseX, mouseY)
    if b>0.92:
        text('Click')
        pyautogui.mouseDown()
        pyautogui.moveTo(mouseX, mouseY)

    if none >0.9:
        pyautogui.mouseUp()
    

    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(1) == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()