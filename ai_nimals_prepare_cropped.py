import os
import hashlib
from os.path import join
import PIL


os.environ["PATH"] += os.pathsep + os.getcwd()
datasetPath = os.getcwd() + "/Downloads"

def getImgsHashs():
    imgshashes = []
    for (i, classPath) in enumerate(os.listdir(datasetPath)):
        imgDir = join(datasetPath, classPath, "detected")
        for (j, imgPath) in enumerate(os.listdir(imgDir)):
            imgPath = join(imgDir, imgPath)
            md5Hash = hashlib.md5()
            with open(imgPath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5Hash.update(chunk)
                imgshashes.append([imgDir, md5Hash.hexdigest()])
    return imgshashes




def main():
    imgDirs = []
    imgDirs = getImgsHashs();
    print(imgDirs)



if __name__ == "__main__":
    main()
