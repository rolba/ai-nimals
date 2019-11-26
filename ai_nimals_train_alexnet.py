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

# This function builds AlexNet. I am experimenting with different configuration fo this network.
def buildAlexnet(width, height, depth, classes, reg):
        alexnetmodel = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        alexnetmodel.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=inputShape, padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Conv2D(256, (5, 5), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(Conv2D(384, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization(axis=chanDim))
        alexnetmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        alexnetmodel.add(Dropout(0.25))

        alexnetmodel.add(Flatten())
        alexnetmodel.add(Dense(4096, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization())
        alexnetmodel.add(Dropout(0.5))

        alexnetmodel.add(Dense(4096, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("sigmoid"))
        alexnetmodel.add(BatchNormalization())
        alexnetmodel.add(Dropout(0.5))

        alexnetmodel.add(Dense(classes, kernel_regularizer=l2(reg)))
        alexnetmodel.add(Activation("softmax"))

        return alexnetmodel

# Data augumentation function. Used to return augumentator object
def augumentator():
    aug = ImageDataGenerator(rotation_range=15, zoom_range=0.1,
                             width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1,
                             horizontal_flip=True, fill_mode="nearest")
    return aug


def main():
    # Concatenate some system paths to datasets in hdf5 databases.
    splittedSetPath = os.getcwd() + "/dataset"
    trainingSetPath = os.path.join(splittedSetPath, "trainset", "trainSet.h5py")
    testingSetPath = os.path.join(splittedSetPath, "testset", "testSet.h5py")
    validatingPath = os.path.join(splittedSetPath, "validationset", "validateSet.h5py")
    labelsPath = os.path.join(splittedSetPath, "labels", "labels.json")
    meansPath = os.path.join(splittedSetPath, "trainset", "train_mean.json")

    # Load train set mean value.
    means = json.loads(open(meansPath).read())
    print(means)

    # Load hdf5 data sets of train, test and validate data
    trainDb = h5py.File(trainingSetPath, "r")
    print(trainDb["labels"].shape[0])
    testDb = h5py.File(testingSetPath, "r")
    print(testDb["labels"].shape[0])
    valDb = h5py.File(validatingPath, "r")
    print(valDb["labels"].shape[0])

    # Build AlexNet model
    alexNetModel = buildAlexnet(width=227, height=227, depth=3, classes=55, reg=0.0001)
    # Prepare optimize object. I will use Adam with different lerning rates. It's Hyper parameter here.
    opt = Adam(lr=1e-4)
    # Now let's compile AlexNet to be digestable by hardware (GPU). This methood prepares model for training.
    alexNetModel.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Mage data generator object. This object is used for generating data for training network.
    # Mind that if you set bathsize too big your training process will be killed without any worning.
    # It's good to monitor your RAM and SWAP file for the first epoch. You should never use SWAP - for more information please refer to blog post!
    trainGenerator = dg(trainDb, batchSize = 32, aug = augumentator(), binarize = True, classesNum=55 )
    # Image resizer
    trainGenerator.setimageresizer(width = 227, height = 227)
    # mean substraction preprocessor
    trainGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])

    # For validation set you want only validation generator and image resizer. No Mean substraction!
    validateGenerator = dg(valDb, batchSize = 32, aug = None, binarize = True, classesNum=55 )
    validateGenerator.setimageresizer(width = 227, height = 227)
    validateGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])

    # Train the network using generated data.
    alexNetModel.fit_generator(
        trainGenerator.generator(),
        steps_per_epoch=trainGenerator.numImages // 32,
        validation_data=validateGenerator.generator(),
        validation_steps=validateGenerator.numImages // 32,
        epochs=20,
        max_queue_size=128 * 2,
        verbose=1,
        workers = 1)
    # save the model to file
    print("[INFO] serializing model...")
    alexNetModel.save(os.path.join(os.getcwd(), "model", "AlexNet.model"), overwrite=True)

if __name__ == "__main__":
    main()


