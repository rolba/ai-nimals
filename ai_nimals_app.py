import numpy as np
import time
import cv2
from imutils.video import FPS
import threading
import queue
import csv
import os
import json
# load the detection module model
print("[INFO] loading detection model...")
detectionNet = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

classesNumber = 22

# initialize the list of birds class labels for VGG16 net the model was trained for
classNamesPath = os.getcwd() + "/dataset"
clPath = os.path.join(classNamesPath, "labels", "labels.json")
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

# classification thread
def classificationWorker(inQueue, outQueue):
    from keras.models import load_model

    # load the birds model
    print("Loading classification model...")
    classModel = load_model(modelPath)

    while True:
        if not inQueue.empty():
            image = inQueue.get()
            image = cv2.resize(image, (224, 224))
            image = np.expand_dims(image, axis=0)
            birdLabel = classModel.predict(image)
            birdLabel = birdLabel[0].astype(np.int32)
            birdLabelPosition = np.where(birdLabel == 1)
            if birdLabelPosition[0]:
                foundClassName = classNames[birdLabelPosition[0][0]]
                outValuesDict = {"birdLabel": birdLabel, "foundClassName": foundClassName}
                outQueue.put(outValuesDict)


# init background process.
print("[INFO] init queues...")
inputQueue = queue.Queue(maxsize=1)
outputQueue = queue.Queue(maxsize=1)

threading.Thread(target=classificationWorker, args=(inputQueue, outputQueue)).start()

# load videofile or load camera interface
print("[INFO] loading video file...")
cap = cv2.VideoCapture('videos/3.mp4')
# cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Init engine and minor values
print("[INFO] initialize engine...")
bbox = (0, 0, 1, 1)
trackerInit = True
trackerReinit = False
kcfTracker = False
nnDetection = True
image = None
birdConfidence = None
detectCntr = 0
foundClassName = ""
time.sleep(1)
fps = FPS().start()
sessionTable = np.zeros(classesNumber)
statisticTable = []
fillQueue = False
queueImage = None
noBird = 0
startTime = time.time()

while (cap.isOpened()):
    fps = FPS().start()
    ret, img = cap.read()
    if not ret:
        print('no image to read')
        break
    img = cv2.resize(img, (640, 480))

    if trackerInit:
        tracker = cv2.TrackerKCF_create()
        ok = tracker.init(img, bbox)
        trackerInit = False

    # proceed bird localization process using mobile net ssd
    if nnDetection:
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

                        # kcf tracker - prepare crop for tracking - to improve algorith speed
                        bbox = (startX, startY, endX, endY)

                        fillQueue = True
                        trackerReinit = True
                        kcfTracker = True
                        nnDetection = False
                        detectCntr = 0
                        break

    # fill queue for object recignition using VGG16 network trained by me
    if fillQueue:
        if inputQueue.empty():
            inputQueue.put(queueImage)
            fillQueue = False

    # if classification is done - take label name
    if not outputQueue.empty():
        detections = outputQueue.get()
        if detections is not None:
            foundClassName = detections["foundClassName"]
            sessionTable += detections["birdLabel"]
    cv2.putText(img, foundClassName + " " + str(birdConfidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                2)

    # kcf tracker - track bird using opencv to speedup algorithm
    if kcfTracker:
        if trackerReinit:
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(img, bbox)
            if ok:
                trackerReinit = False
            else:
                print("[ERROR]: KCF Reinit Fail, detectCntr: ", detectCntr)
                kcfTracker = False
                nnDetection = True
        else:
            ok, newbox = tracker.update(img)

            if ok:
                noBird = 0
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(img, p1, p2, (10, 255, 100), 2)

            else:
                print("[ERROR]: KCF Lost, detectCntr: ", detectCntr)
                kcfTracker = False
                nnDetection = True

    if detectCntr == 3:
        nnDetection = True
        kcfTracker = False
        detectCntr = 0

    detectCntr += 1
    noBird += 1

    # statistic update
    if noBird == 3:
        if (sessionTable == 0).all():
            noBird = 0
        else:
            ts = time.time()
            local = []
            statisticTable.append([classNames[sessionTable.argmax()], str(int(ts))])

            noBird = 0
            sessionTable = np.zeros(classesNumber)

    fps.update()
    fps.stop()
    cv2.putText(img, "FPS: " + str(fps.fps()), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    detections = None

    nowTime = time.time()
    if nowTime - startTime > 20:
        startTime = time.time()
        sep = ","
        with open('statistic.csv', 'a') as f:
            writer = csv.writer(f)
            for row in statisticTable:
                print(row)
                writer.writerow(row)

        statisticTable = []

cap.release()
cv2.destroyAllWindows()



