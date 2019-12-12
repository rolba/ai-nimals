import numpy as np
import time
import cv2
from imutils.video import FPS
import os
import json
from keras.models import load_model
print("[INFO] loading detection model...")
detectionNet = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

classesNumber = 22

# Initialize paths for labels json
classNamesPath = os.getcwd() + "/dataset"
clPath = os.path.join(classNamesPath, "labels", "labels_short.json")
labels = json.loads(open(clPath).read())
classNames = labels["labels"]

# Initialize the list of class labels for MobileNet SSD the model was trained for
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize model path, I got my VGG16 finetuned model here.
modelsPath = os.getcwd() + "/model"
modelPath = os.path.join(modelsPath, "ai_nimals_finetuned_vgg16.model")

# Load video file for tests
cap = cv2.VideoCapture('videos/3.mp4')
time.sleep(2.0)

# Init minor values
print("[INFO] initialize engine...")
birdConfidence = None
foundClassName = ""
predict = False
detectionText = ""
birdImage=None
startTime = time.time()

# Mean Init. Yeah I had it saved some on my HDD
mean = {"B": 102.46804193022903, "R": 127.70118085492695, "G": 122.15603763839495}

#Load model
classModel = load_model(modelPath)

#Start looping over video file
while (cap.isOpened()):
    #I need this to measure FPS
    fps = FPS().start()

    #Read one frame
    ret, img = cap.read()
    if not ret:
        print('no image to read')
        break

    # Resizing can be important on Jetson platform. On my GTX1060 I don't have to think about it
    #img = cv2.resize(img, (640, 480))

    # proceed bird localization process using mobile net ssd
    (h, w) = img.shape[:2]

    # Perform mean subtraction and scalling using blobFromImage function
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)

    # Pass blob through Mobile net SSD network
    detectionNet.setInput(blob)
    detections = detectionNet.forward()
    blob = None

    # if detections are found - proceed
    if detections.shape[2] > 0:
        # Iterate over detections
        for i in np.arange(0, detections.shape[2]):
            birdConfidence = detections[0, 0, i, 2]
            # proceed if object was detected with confidence bigger than 0.4
            if birdConfidence > 0.4:
                idx = int(detections[0, 0, i, 1])
                # Check if detected object is a bird - proceed
                if CLASSES[idx] == 'bird':
                    # Prepare boundary boxes of a bird
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

                    # Draw boundary boxes around the bird
                    cv2.rectangle(img, (startX, startY), (endX, endY), (10, 255, 100), 2)

                    # Crop bird detection ROI and prepare to classify it by a classificator
                    birdImage = img[startY:startY + (endY - startY), startX:startX + (endX - startX)]

                    predict = True
                    break

    if predict:
        # Prepare image for classification. First resize, than
        birdImage = cv2.resize(birdImage, (224, 224))

        # Mean subtract
        (B, G, R) = cv2.split(birdImage.astype("float32"))
        R = R - mean["B"]
        G = G - mean["G"]
        B = B - mean["R"]
        birdImage = cv2.merge([B, G, R])
        birdImage = np.expand_dims(birdImage, axis=0)

        # Push cropped image through the network
        birdLabel = classModel.predict(birdImage)

        # Get the label number
        birdLabel = birdLabel[0].astype(np.int32)
        birdLabelPosition = np.where(birdLabel == 1)

        # Find label name
        if birdLabelPosition[0]:
            foundClassName = classNames[birdLabelPosition[0][0]]
            print(birdLabel, foundClassName)

        predict = False

    # Put some text on the image
    cv2.putText(img, foundClassName + " " + str(birdConfidence), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Count Frames per second to monitor performance.
    fps.update()
    fps.stop()
    cv2.putText(img, "FPS: " + str(fps.fps()), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Finally thow the imagae with all information
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    detections = None

cap.release()
cv2.destroyAllWindows()
