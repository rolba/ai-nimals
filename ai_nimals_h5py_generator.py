import h5py
import os
import pandas as pd
from numpy import single

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
        dataSetCsv = pd.read_csv(singleSetPath)

        print(dataSetCsv.to)
        # for p in dataSetCsv:
            # print(p)
            # label = p.split(os.path.sep)[-2]
            # print(label)
            # label.append(label)

    print (dataSet)


if __name__ == "__main__":
    main()