import h5py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import json

os.environ["PATH"] += os.pathsep + os.getcwd()

# Discover datatset path and construct it
datasetPath = os.getcwd() + "/Downloads"
splittedSetPath = os.getcwd() + "/dataset"

# I need paths to save images paths to CSV

trainingSetPath = os.path.join(splittedSetPath, "trainset")
testingSetPath = os.path.join(splittedSetPath, "testset")
validatingPath = os.path.join(splittedSetPath, "validationset")

dataSet = []
dataSet.append(os.path.join(trainingSetPath, "trainSet.csv"))
dataSet.append(os.path.join(testingSetPath, "testSet.csv"))
dataSet.append(os.path.join(validatingPath, "validateSet.csv"))

def main():
    (R, G, B) = ([], [], [])

    for singleSetPath in dataSet:
        setFileName = singleSetPath.split(os.path.sep)[-1]
        dataPaths = pd.read_csv(singleSetPath).values
        le = LabelEncoder()
        labels = []
        for p in dataPaths:
            label = p[0].split(os.path.sep)[-3]
            labels.append(label)
        labels = le.fit_transform(labels)

        for (i, (path, label)) in enumerate(zip(dataPaths, labels)):
            # load the image and process it
            image = cv2.imread(path[0])
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            if setFileName == "trainSet.csv":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)

    D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
    f = open(os.path.join(trainingSetPath, "trainmean.json"), "w")
    f.write(json.dumps(D))
    f.close()


if __name__ == "__main__":
    main()