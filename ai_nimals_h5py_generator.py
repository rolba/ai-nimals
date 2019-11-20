from DatasetWriter import DatasetWriter
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import json

os.environ["PATH"] += os.pathsep + os.getcwd()

# Discover datatset path and construct it.
datasetPath = os.getcwd() + "/Downloads"
# Discover dataset splited csv files.
splittedSetPath = os.getcwd() + "/dataset"

# Construct training, validating and testing csv files containing paths to images
trainingSetPath = os.path.join(splittedSetPath, "trainset")
testingSetPath = os.path.join(splittedSetPath, "testset")
validatingPath = os.path.join(splittedSetPath, "validationset")
labelsPath = os.path.join(splittedSetPath, "labels")
# Create list for iterating over datasets.
dataSet = []
dataSet.append(os.path.join(trainingSetPath, "trainSet.csv"))
dataSet.append(os.path.join(testingSetPath, "testSet.csv"))
dataSet.append(os.path.join(validatingPath, "validateSet.csv"))

classes = []
# Discover and update labels list with all labels in dataset
dirList = os.listdir(datasetPath)
for p in dirList:
    classes.append(p[:-5])
f = open(labelsPath+"/labels.json", "w")
f.write(json.dumps({"labels": classes}))
f.close()

def main():
    # create 3 lists that will acumulate mean of all training images. This will be needed in training process.
    (R, G, B) = ([], [], [])

    # Iterate over all datasets csv files' paths
    for singleSetPath in dataSet:
        # Split single csv data set and extract csv filename
        setFileName = singleSetPath.split(os.path.sep)[-1]

        print(setFileName)
        # Read CSV file to memory
        dataPaths = pd.read_csv(singleSetPath).values
        # Create LabelEncoder object for encoding labels or classes
        le = LabelEncoder()
        labels = []

        # Discover and update labels list with all labels in dataset
        for p in dataPaths:
            label = p[0].split(os.path.sep)[-3]
            labels.append(label)
        # transform labels' list to numeric list
        labels = le.fit_transform(labels)

        # Create h5py file container object
        # Name of h5py file is my CSV file without extension
        # Dimenions of h5py dataset should fit into images quantity and their resolution
        writer = DatasetWriter(singleSetPath[:-3]+"h5py", (len(dataPaths), 256, 256, 3))

        # Interate over all images in csv dataset
        for (i, (path, label)) in enumerate(zip(dataPaths, labels)):
            # load the image
            image = cv2.imread(path[0])

            # resize image to 256x256 pixels
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

            # write image to h5py file container
            writer.add(image, label, i)

            # append r g b channel mean of certain image to list of means
            if setFileName == "trainSet.csv":
                (b, g, r) = cv2.mean(image)[:3]
                R.append(r)
                G.append(g)
                B.append(b)
        # When iterating over single csv data set is finished, close h5py file.
        writer.close()

    # count mean of R, G, B channels separatly
    D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}

    # Serialize means to json file
    f = open(os.path.join(trainingSetPath, "train_mean.json"), "w")
    f.write(json.dumps(D))
    f.close()

if __name__ == "__main__":
    main()


