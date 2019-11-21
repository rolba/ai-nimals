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
from DataGenerator import DataGenerator as dg
import numpy as np
import json
import os
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

def augumentator():
    aug = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
                             width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")
    return aug


def main():
    splittedSetPath = os.getcwd() + "/dataset"
    trainingSetPath = os.path.join(splittedSetPath, "trainset", "trainSet.h5py")
    testingSetPath = os.path.join(splittedSetPath, "testset", "testSet.h5py")
    validatingPath = os.path.join(splittedSetPath, "validationset", "validateSet.h5py")
    labelsPath = os.path.join(splittedSetPath, "labels", "labels.json")
    meansPath = os.path.join(splittedSetPath, "trainset", "train_mean.json")

    means = json.loads(open(meansPath).read())
    print(means)
    trainDb = h5py.File(trainingSetPath, "r")
    print(trainDb["labels"].shape[0])
    testDb = h5py.File(testingSetPath, "r")
    print(testDb["labels"].shape[0])
    valDb = h5py.File(validatingPath, "r")
    print(valDb["labels"].shape[0])

    alexNetModel = buildAlexnet(width=227, height=227, depth=3, classes=55, reg=0.0003)
    opt = Adam(lr=1e-3)
    alexNetModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    trainGenerator = dg(trainDb, batchSize = 128, aug = augumentator(), binarize = True, classesNum=55 )
    trainGenerator.setimageresizer(width = 227, height = 227)
    trainGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])

    validateGenerator = dg(valDb, batchSize = 128, aug = augumentator(), binarize = True, classesNum=55 )
    validateGenerator.setimageresizer(width = 227, height = 227)
    validateGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])


    # train the network
    alexNetModel.fit_generator(
        trainGenerator.generator(),
        steps_per_epoch=trainGenerator.numImages // 128,
        validation_data=validateGenerator.generator(),
        validation_steps=validateGenerator.numImages // 128,
        epochs=200,
        max_queue_size=128 * 2,
        verbose=1)

if __name__ == "__main__":
    main()


