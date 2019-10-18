import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from os.path import join
import cv2
import numpy as np
import shutil

os.environ["PATH"] += os.pathsep + os.getcwd()

def getBwLittleImgs(datasetPath):
    # Find all classes paths in directory and iterate over it
    for (i, classPath) in enumerate(os.listdir(datasetPath)):

        # Construct cropped images dir
        imgDir = join(datasetPath, classPath, "detected")
        bwDir = join(datasetPath, classPath, "bwdir")
        print(classPath)
        # Create Donwload patch or delete existing!
        if not os.path.exists(bwDir):
            os.makedirs(bwDir)
        else:
            shutil.rmtree(bwDir)
            os.makedirs(bwDir)

        # Iterate over all images in directory
        for (j, imgName) in enumerate(os.listdir(imgDir)):

            # Construct image patch
            imgPath = join(imgDir, imgName)

            image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, (32, 32))
                resized_image = np.array(resized_image)
                cv2.imwrite(os.path.join(bwDir, imgName), resized_image)
            else:
                print(imgPath)
                os.remove(imgPath)


def deleteDupl(imgPath):
    os.remove(imgPath)

def findDelDuplBw(searchedName, bwDir):
        searchedImg = join(bwDir, searchedName)

        for (j, cmpImageName) in enumerate(os.listdir(bwDir)):
            if cmpImageName == searchedName:
                pass
            else:
                cmpImageBw = join(bwDir, cmpImageName)

                try:
                    searchedImageBw = np.array(cv2.imread(searchedImg, cv2.IMREAD_GRAYSCALE))
                    cmpImage = np.array(cv2.imread(cmpImageBw, cv2.IMREAD_GRAYSCALE))
                    rms = sqrt(mean_squared_error(searchedImageBw, cmpImage))
                except:
                    continue

                if rms < 3:
                    os.remove(cmpImageBw)
                    print (searchedImg, cmpImageName, rms)

def findDelDetected(detectedDir, bwDir):
    bwFiles = os.listdir(bwDir)
    for file in os.listdir(detectedDir):
        if file not in bwFiles:
            print (file,  " to be deleted")
            os.remove(os.path.join(detectedDir, file))

def main():

    # Define working directory - direcotry with our dowloaded data images
    datasetPath = os.getcwd() + "/Downloads"

    # To clean data I wan to produce 32x32 pix images of data set.
    # And store them in "bwdir" in every class
    getBwLittleImgs(datasetPath)

    # Now lets iterate over all classes in data set
    for (i, classPath) in enumerate(os.listdir(datasetPath)):

        # Join detected by previous script path
        detectedDir = join(datasetPath, classPath, "detected")
        # Join black-white images path
        bwDir = join(datasetPath, classPath, "bwdir")

        # Iterate over images in one class - detected images previously by MobileSSD net
        for (i, detectedImg) in enumerate(os.listdir(detectedDir)):
            # Find duplicated BW images and delete duplicates.
            findDelDuplBw(detectedImg, bwDir)

        # Basing on cleaned BW images, now clean detected direcotry comparing data
        # between detected directory and bwDir directory
        findDelDetected(detectedDir, bwDir)

        # Remove bwDir - we don't need it any more
        shutil.rmtree(bwDir)



if __name__ == "__main__":
    main()
