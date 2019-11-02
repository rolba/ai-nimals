import h5py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    for singleSetPath in dataSet:
        setFileName = singleSetPath.split(os.path.sep)[-1]
        dataSetCsv = pd.read_csv(singleSetPath).values
        le = LabelEncoder()
        labels = []
        for p in dataSetCsv:
            label = p[0].split(os.path.sep)[-3]
            labels.append(label)
        labels = le.fit_transform(labels)


if __name__ == "__main__":
    main()