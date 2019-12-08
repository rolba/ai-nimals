import numpy as np
import time
import cv2
from imutils.video import FPS
import queue
import os
import json
from keras.models import load_model
print("[INFO] loading detection model...")
detectionNet = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

classesNumber = 22

# initialize the list of birds class labels for VGG16 net the model was trained for
classNamesPath = os.getcwd() + "/dataset"
clPath = os.path.join(classNamesPath, "labels", "labels_short.json")
labels = json.loads(open(clPath).read())
classNames = labels["labels"]

# initialize the list of class labels for MobileNet SSD the model was trained for
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

modelsPath = os.getcwd() + "/model"
modelPath = os.path.join(modelsPath, "ai_nimals_finetuned_vgg16.model")

# init background process.
print("[INFO] init queues...")
inputQueue = queue.Queue(maxsize=1)
outputQueue = queue.Queue(maxsize=1)

# load videofile or load camera interface
print("[INFO] loading video file...")
cap = cv2.VideoCapture('videos/3.mp4')
time.sleep(2.0)

# Init engine and minor values
print("[INFO] initialize engine...")
birdConfidence = None
foundClassName = ""
predict = False
startTime = time.time()
classModel = load_model(modelPath)
detectionText = ""

while (cap.isOpened()):
    fps = FPS().start()
    ret, img = cap.read()
    if not ret:
        print('no image to read')
        break
    img = cv2.resize(img, (640, 480))

    # proceed bird localization process using mobile net ssd

    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

    detectionNet.setInput(blob)
    blob = None
    detections = detectionNet.forward()
    # if detections are found - proceed
    if detections.shape[2] > 0:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # proceed if object was detected with confidence bigger than 0.4
            if confidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                # if detected object is a bird - proceed
                if CLASSES[idx] == 'bird':
                    noBird = 0
                    birdConfidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    if startX < 0:
                        startX = 0
                    if startY < 0:
                        startY = 0
                    if endX > 640:
                        endX = 640
                    if endY > 480:
                        endY = 480

                    # crop bird detection ROI
                    queueImage = img[startY:startY + (endY - startY), startX:startX + (endX - startX)]
                    cv2.rectangle(img, (startX, startY), (endX, endY), (10, 255, 100), 2)

                    predict = True
                    break

    if predict:
        queueImage = cv2.resize(queueImage, (224, 224))
        queueImage = np.expand_dims(queueImage, axis=0)
        birdLabel = classModel.predict(queueImage)

        birdLabel = birdLabel[0].astype(np.int32)
        birdLabelPosition = np.where(birdLabel == 1)
        if birdLabelPosition[0]:
            foundClassName = classNames[birdLabelPosition[0][0]]
            print(birdLabel, foundClassName)

        predict = False

    cv2.putText(img, foundClassName + " " + str(birdConfidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    fps.update()
    fps.stop()
    cv2.putText(img, "FPS: " + str(fps.fps()), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    detections = None

cap.release()
cv2.destroyAllWindows()