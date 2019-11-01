import os
import random
import shutil
import pandas

os.environ["PATH"] += os.pathsep + os.getcwd()

# Typically data set can be devided in to 3 parts
# Training set part which can be 60-80% of orginal data set quantity
trainingSetRatio = 0.75
# Validation set part which can be 10-20% of orginal data set quantity
# Tis set us used to check how your training process perform.
validationSetRatio = 0.15
# Testing set part which can be 10-20% of orginal data set quantity
# You check how your model performs on unseen data when applying trained model
testingSetRatio = 0.1

def makeDelPath(workingPath):
    # Check if destination folder exists. If yes - delete it. If no - create.
    if not os.path.exists(workingPath):
        os.makedirs(workingPath)
    else:
        shutil.rmtree(workingPath)
        os.makedirs(workingPath)

def main():
    # Discover datatset path and construct it
    datasetPath = os.getcwd() + "/Downloads"
    splittedSetPath = os.getcwd() + "/dataset"

    # I need paths to save images paths to CSV
    trainingSetPath = os.path.join(splittedSetPath, "trainset")
    testingSetPath = os.path.join(splittedSetPath, "testset")
    validatingPath = os.path.join(splittedSetPath, "validationset")

    makeDelPath(trainingSetPath)
    makeDelPath(testingSetPath)
    makeDelPath(validatingPath)

    # List of file paths for training
    trainFileList = []
    # List of file paths for testsing
    testfileList = []
    # List of file paths for validating
    validateFileList = []

# Iterate over this directory. classFolder is a foilder that has images fo given class
    for (i, classFolder) in enumerate (os.listdir(datasetPath)):

        # Concatenate path with images to be splitted
        datasetDetectedClass = os.path.join(datasetPath, classFolder, "detected")
        # List all images in this direcotry
        dataSetList = os.listdir(datasetDetectedClass)
        # Count quantity of images
        calssImagesQuantity = len(dataSetList)
        # I have to get rid of "_bird" from calssname. It can be hardcoded.
        className = classFolder[:-5]

        # Calculate quantity files for all data sets
        validatingSetQuantity = int(calssImagesQuantity * validationSetRatio-1)
        testingSetQuantity = int(calssImagesQuantity * testingSetRatio-1)
        trainingSetQuantity = int(calssImagesQuantity * trainingSetRatio)
        trainingSetQuantity += (calssImagesQuantity - (trainingSetQuantity+validatingSetQuantity+testingSetQuantity))

        # Shuffle dataSetList
        random.shuffle(dataSetList)
        random.shuffle(dataSetList)
        random.shuffle(dataSetList)

        # Create sub list of files for traiing set
        trainingSet = dataSetList[0:trainingSetQuantity]
        # Create list of paths
        for element in trainingSet:
            trainFileList.append(str(os.path.join(datasetDetectedClass, element)))

        # Create sub list of files for testing set
        testingSet = dataSetList[trainingSetQuantity:testingSetQuantity + trainingSetQuantity]
        # Create list of paths
        for element in testingSet:
            testfileList.append(str(os.path.join(datasetDetectedClass, element)))

        # Create sub list of files for validation set
        validationSet = dataSetList[testingSetQuantity + trainingSetQuantity: testingSetQuantity + trainingSetQuantity + validatingSetQuantity]
        # Create list of paths
        for element in validationSet:
            validateFileList.append(str(os.path.join(datasetDetectedClass, element)))


    # Shuffle image path list
    random.shuffle(trainFileList)
    random.shuffle(trainFileList)
    random.shuffle(trainFileList)
    # Save to CSV
    df = pandas.DataFrame(trainFileList)
    df.to_csv(os.path.join(trainingSetPath, "trainSet.csv"), sep=',', index=False)

    # Shuffle image path list
    random.shuffle(testfileList)
    random.shuffle(testfileList)
    random.shuffle(testfileList)
    # Save to CSV
    df = pandas.DataFrame(testfileList)
    df.to_csv(os.path.join(testingSetPath, "testSet.csv"), sep=',', index=False)

    # Shuffle image path list
    random.shuffle(validateFileList)
    random.shuffle(validateFileList)
    random.shuffle(validateFileList)
    # Save to CSV
    df = pandas.DataFrame(validateFileList)
    df.to_csv(os.path.join(validatingPath, "validateSet.csv"), sep=',', index=False)


if __name__ == "__main__":
    main()