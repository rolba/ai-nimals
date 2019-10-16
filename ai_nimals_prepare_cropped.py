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
        for (j, imgPath) in enumerate(os.listdir(imgDir)):

            # Construct image patch
            imgPath = join(imgDir, imgPath)

            image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                resized_image = cv2.resize(image, (32, 32))
                resized_image = np.array(resized_image)
                cv2.imwrite(os.path.join(bwDir, str(j)+'.jpg'), resized_image)
            else:
                print(imgPath)
                os.remove(imgPath)


def cleanDuplImgs(datasetPath, rmsErrorThreshold = 5):

    for (i, classPath) in enumerate(os.listdir(datasetPath)):
        imgDir = join(datasetPath, classPath, "detected")
        for (j, imgName) in enumerate(os.listdir(imgDir)):
            imgPath = join(imgDir, imgName)
            if os.path.exists(imgPath):
                image = np.array(cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE))

                for (k, cmpImgName) in enumerate(os.listdir(imgDir)):
                    if imgName == cmpImgName:
                        pass
                    else:
                        cmpImgPath = join(imgDir, cmpImgName)
                        if os.path.exists(cmpImgPath):
                            cmpImage = np.array(cv2.imread(cmpImgPath, cv2.IMREAD_GRAYSCALE))

                            rms = sqrt(mean_squared_error(image, cmpImage))
                            cor = np.corrcoef(image.flat, cmpImage.flat)

                            if rms < rmsErrorThreshold:
                                os.remove(cmpImgPath)
                                print (imgName, cmpImgName, rms, cor[0, 1])
                        else:
                            pass
            else:
                pass

def main():
    datasetPath = os.getcwd() + "/Downloads"

    # getBwLittleImgs(datasetPath)
    cleanDuplImgs(datasetPath)



if __name__ == "__main__":
    main()
