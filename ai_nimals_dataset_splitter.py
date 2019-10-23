import os
import random
import shutil


os.environ["PATH"] += os.pathsep + os.getcwd()

# Typically data set can be devided in to 3 parts
# Training set part which can be 60-70% of orginal data set quantity
trainingSetRatio = 0.6
# Validation set part which can be 10-20% of orginal data set quantity
# Tis set us used to check how your training process perform.
validationSetRatio = 0.2
# Testing set part which can be 10-20% of orginal data set quantity
# You check how your model performs on unseen data when applying trained model
testingSetRatio = 0.2

def fileCopier(sourcePath, destPath, setFiles):
    print("source:", sourcePath, "dest:", destPath, len(setFiles))

    if not os.path.exists(destPath):
        os.makedirs(destPath)
    else:
        shutil.rmtree(destPath)
        os.makedirs(destPath)

    for fileName in setFiles:
        fileToCopyPath = os.path.join(sourcePath, fileName)
        shutil.copy(fileToCopyPath, destPath)


def main():
    # Discover datatset path and construct it
    datasetPath = os.getcwd() + "/Downloads"
    
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
        
        
        validationSetQuantity = int(calssImagesQuantity * validationSetRatio-1)
        testingSetQuantity = int(calssImagesQuantity * testingSetRatio-1)
        trainingSetQuantity = int(calssImagesQuantity * trainingSetRatio)
        trainingSetQuantity += (calssImagesQuantity - (trainingSetQuantity+validationSetQuantity+testingSetQuantity))

        random.shuffle(dataSetList)
        random.shuffle(dataSetList)
        random.shuffle(dataSetList)

        trainingSet = dataSetList[0:trainingSetQuantity]
        testingSet = dataSetList[trainingSetQuantity:testingSetQuantity+trainingSetQuantity]
        validationSet = dataSetList[testingSetQuantity+trainingSetQuantity: testingSetQuantity+trainingSetQuantity+validationSetQuantity]

        trainingSetPath = os.path.join(datasetPath, classFolder, "trainset")
        testingSetPath = os.path.join(datasetPath, classFolder, "testset")
        validationPath = os.path.join(datasetPath, classFolder, "validationset")

        fileCopier(datasetDetectedClass, trainingSetPath, trainingSet)
        fileCopier(datasetDetectedClass, testingSetPath, testingSet)
        fileCopier(datasetDetectedClass, validationPath, validationSet)


if __name__ == "__main__":
    main()
