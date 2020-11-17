
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO
import time


#from threading import Thread    # нужно для улучшений

delay = 1
mask_detect = False



def detect_and_predict_mask(frame, faceNet, maskNet):

    #возьмем кадр из потокового видеопотока и изменим его размер
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))


    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print(detections.shape)

    # инициализировать наш список лиц, соответствующий им вектор и вектор сети
    faces = []
    locs = []
    preds = []

    # цикл определения лиц
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# загрузка модели и вспомогательных файлов
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

# iинициализация видеопотока
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()



def video_stream_recognise():
    global mask_detect
    global angle
    while True:
        #возьмем кадр из потокового видеопотока и изменим его размер, на максимальную ширину 400 пикселей
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # отпределение лиц
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # цмкл по определеннм лицам
        for (box, pred) in zip(locs, preds):
            
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            if (mask > withoutMask):
                print("mask")
                label = "Mask"
                mask_detect = True
            else:
                print("no Mask")
                label = "NO Mask"
                mask_detect = False
            
            if mask_detect:
                forward(int(delay) / 1000.0, int(120))
                time.sleep(3)
                backwards(int(delay) / 1000.0, int(120))
                                
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # добавляем подпись к рамке
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
	
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # показать кадр
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
coil_A_1_pin = 17 # IN1
coil_A_2_pin = 18 # IN2
coil_B_1_pin = 21 # IN3
coil_B_2_pin = 22 # IN4

# настройка подачи напряжения на обмотки катушки
StepCount=8
Seq = [[1,0,0,1],
       [1,0,0,0],
       [1,1,0,0],
       [0,1,0,0],
       [0,1,1,0],
       [0,0,1,0],
       [0,0,1,1],
       [0,0,0,1]]


GPIO.setup(coil_A_1_pin, GPIO.OUT)
GPIO.setup(coil_A_2_pin, GPIO.OUT)
GPIO.setup(coil_B_1_pin, GPIO.OUT)
GPIO.setup(coil_B_2_pin, GPIO.OUT)


def setStep(w1, w2, w3, w4):
    GPIO.output(coil_A_1_pin, w1)
    GPIO.output(coil_A_2_pin, w2)
    GPIO.output(coil_B_1_pin, w3)
    GPIO.output(coil_B_2_pin, w4)

def forward(delay, steps):    # поворот против часовой стрелки
    for i in range(steps):
        for j in range(StepCount):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)

def backwards(delay, steps):   # поворот по часовой стрелке
    for i in range(steps):
        for j in reversed(range(StepCount)):
            setStep(Seq[j][0], Seq[j][1], Seq[j][2], Seq[j][3])
            time.sleep(delay)

		
		
if __name__ == '__main__':
    video_stream_recognise()




# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()




'''

улучшение кода (возможность закрывать дверь если человек снял маску проходя через дверь)

эта часть слишком тяжелая для raspberry pi 3 (работает на intel i3 ) , возможно, заработает на RPi4


def opendoorfunc():
    global angle
    while True:
        time.sleep(0.5)
        if (mask_detect):
            while (angle != 90):
                if mask_detect:
                    angle += 10
                    print("open, angle = " + str(angle))
                    time.sleep(0.5)
                else:
                    while (angle != 0):
                        angle -= 10
                        print("close, angle = " + str(angle))
                        time.sleep(0.5)
        else:
            while (angle != 0):
                angle -= 10
                print("close, angle = " + str(angle))
                time.sleep(0.5)
        #...    


thread1 = Thread(target=video_stream_recognise, args=())
thread2 = Thread(target=opendoorfunc , args=())
 
thread1.start()
thread2.start()
thread1.join()
thread2.join()
'''
