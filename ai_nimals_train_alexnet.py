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

import cv2

def buildAlexnet():
    def build(width, height, depth, classes, reg):
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
    pass


if __name__ == "__main__":
    main()


