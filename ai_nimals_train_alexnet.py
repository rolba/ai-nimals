# USAGE
# python train_alexnet.py

# --checkpoints output/adam2/checkpoints --model output/adam2/checkpoints/epoch_5.hdf5 --start-epoch 5

# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import ai_nimals_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor, EpochCheckpoint
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet, DeeperGoogLeNet, ResNet
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.optimizers import Adam, SGD
from keras.models import load_model
import json
import argparse
import keras.backend as K

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
    help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
    help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
    help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(224,224)
pp = PatchPreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 16, aug=aug,
    preprocessors=[pp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 16,
    preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 16, preprocessors=[pp, iap], classes=config.NUM_CLASSES)
if args["model"] is None:
    print("[INFO] compiling model...")
#     opt = Adam(lr=1e-3)
#     model = AlexNet.build(width=227, height=227, depth=3, classes=config.NUM_CLASSES, reg=0.0003)
#     model = DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0005)
    opt = SGD(lr=1e-1)
    model = ResNet.build(224, 224, 3, 6, (2, 2, 2, 2),(32, 128, 256, 512, 128), reg=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-1)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))


callbacks = [
    EpochCheckpoint(args["checkpoints"], every=5,
        startAt=args["start_epoch"]),
    TrainingMonitor(config.OUTPUT_PATH+"/alex_net_ai_nimals.png",
        jsonPath=config.OUTPUT_PATH+"/alex_net_ai_nimals.json",
        startAt=args["start_epoch"])]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 16,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 16,
    epochs=100,
    max_queue_size=16 * 2,
    callbacks=callbacks, verbose=1)

# save the model to file
print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // 16, max_queue_size=16 * 2)
# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))
# close the HDF5 datasets
trainGen.close()
valGen.close()