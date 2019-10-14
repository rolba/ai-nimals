import os
import hashlib
from os.path import join
import cv2

os.environ["PATH"] += os.pathsep + os.getcwd()


# Function finds matching images by it's hash
def getImgsHashs(datasetPath):
    imgs = []
    hashes = []

    # Find all classes paths in directory and iterate over it
    for (i, classPath) in enumerate(os.listdir(datasetPath)):

        # Construct cropped images dir
        imgDir = join(datasetPath, classPath, "detected")

        # Iterate over all images in directory
        for (j, imgPath) in enumerate(os.listdir(imgDir)):

            # Construct image patch
            imgPath = join(imgDir, imgPath)

            image = cv2.imread(imgPath)
            resized_image = cv2.resize(image, (32, 32))
            print(os.path.join(imgDir, str(j)+'_a.jpg'))
            cv2.imwrite(os.path.join(imgDir, str(j)+'_a.jpg'), resized_image)

            # Prepare new md5 object from hashlib
            md5Hash = hashlib.md5()

            # Open image
            with open(imgPath, 'rb') as f:

                # Start iterate over all bytes in image (read every 4096 bytes)
                for chunk in iter(lambda: f.read(32), b""):

                    # Update hash every 4096 bytes
                    md5Hash.update(chunk)

                # Append hash and image patch to lists
                imgs.append([imgPath])
                hashes.append(md5Hash.hexdigest())

    # Return lists of hashes and image paths. Those lists are aligned.
    return imgs, hashes

# Function deletes images that are duplicated
def delDuplicates(im, hs):
    imgs = im
    hashs = hs

    # Iterate over all images in detected directory
    for i, img in enumerate(imgs):
        hs = hashs[i]

        # If image's hash duplicates.
        if hashs.count(hs)>1:

            # Remove them from directory and from hashes and image directory lists
            # os.remove(img[0])
            del(hashs[i])
            del(imgs[i])
            print("deleted", imgs[i])
        else:
            print("NOPE", imgs[i])
    return imgs, hashs


def main():
    datasetPath = os.getcwd() + "/Downloads"

    # Calculate images hashes and prepare aligned lists of hashes and image paths
    imgs, hashs = getImgsHashs(datasetPath)

    # Delete duplicates and prepare new lists without duplicates
    imgs, hashs = delDuplicates(imgs, hashs)

    # Check if lists are the same lenght
    if len(imgs) == len(hashs):
        print("Finished - All Ok")
    else:
        print("Something went wrong. Lists are not the same length")



if __name__ == "__main__":
    main()
