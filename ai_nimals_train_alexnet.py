from sklearn.feature_extraction.image import extract_patches_2d

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras import backend as K

import os
import cv2
import h5py

def buildAlexnet(width, height, depth, classes, reg):
        alexnetmodel = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        alexnetmodel.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Flatten())
        alexnetmodel.add(Dense(4096, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization())
        alexnetmodel.add(Dropout(0.5))

        alexnetmodel.add(Dense(4096, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("relu"))
        alexnetmodel.add(BatchNormalization())
        alexnetmodel.add(Dropout(0.5))

        alexnetmodel.add(Dense(classes, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("softmax"))

        return alexnetmodel

def meanPreprocessor(image, r, g, b):
    (B, G, R) = cv2.split(image.astype("float32"))
    R = R - r
    G = G - g
    B = B - b
    return cv2.merge([B, G, R])

def resizePreprocessor(image, h, w, inter=cv2.INTER_AREA):
    return cv2.resize(image, (h, w), interpolation=inter)

def imgToArray(image, dataFormat = None):
    return img_to_array(image, data_format=dataFormat)

def augumentator():
    aug = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
                             width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")
    return aug


def generator():
    pass

def main():
    splittedSetPath = os.getcwd() + "/dataset"
    trainingSetPath = os.path.join(splittedSetPath, "trainset", "trainSet.h5py")
    testingSetPath = os.path.join(splittedSetPath, "testset", "testSet.h5py")
    validatingPath = os.path.join(splittedSetPath, "validationset", "validateSet.h5py")
    labelsPath = os.path.join(splittedSetPath, "labels", "labels.json")

    trainDb = h5py.File(trainingSetPath, "r")
    print(trainDb["labels"].shape[0])
    testDb = h5py.File(testingSetPath, "r")
    print(testDb["labels"].shape[0])
    valDb = h5py.File(validatingPath, "r")
    print(valDb["labels"].shape[0])


    alexNetModel = buildAlexnet(width=227, height=227, depth=3, classes=55, reg=0.0003)
    opt = Adam(lr=1e-3)
    alexNetModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])





if __name__ == "__main__":
    main()


