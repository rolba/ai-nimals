from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
# Usage:
#     trainGenerator = DataGenerator(hdf5_daatabase, batchSize = 32, aug = augumentator(), binarize = True, classesNum=4 )
#     trainGenerator.setimageresizer(width = 227, height = 227)
#     trainGenerator.setmeanpreprocessor(means["R"], means["G"], means["B"])
class DataGenerator:

	# Constuctor - opens and creates data generator object.
	# @param dbObj - hfg5 ddata base. it must have ["images"] and ["labels"]
	# @param bathSize - size of dat bath. It should be set not to reach your SWAP partition. Otherweis using generator is insane!
	# @param binarize - should be set fot true if you are using categorical_crossentropy loss function - please refer to Kears documantation
	# @param classesNum - number of your labels, classes or cats speccious...
	def __init__(self, dbObj, batchSize, aug=None, binarize=True, classesNum=2):
		self.batchSize = batchSize
		self.aug = aug
		self.binarize = binarize
		self.classesNum = classesNum
		self.db = dbObj
		self.numImages = self.db["labels"].shape[0]
		self.mean = None
		self.width = None
		self.height = None
		self.inter = None

	# Method sets resizer preprocessor.
	# @param width - Width of final image
	# @param height - Height of final image
	# @param inter - Interpolation methood
	def setimageresizer(self, width, height, inter = cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter

	# Method sets mean preprocessor.
	# @param r,g,b - Read, Green Blue channel mean of training dataset.
	def setmeanpreprocessor(self, r, g, b):
		self.mean = [r,g,b]

	# Method used for generating data for training.
	# @param passes - Numer of training epochs.
	def generator(self, passes=np.inf):
		epochs = 0

		# Iterate over epochs
		while epochs < passes:
			for i in np.arange(0, self.numImages, self.batchSize):
				# Take sublist of images and albels
				images = self.db["images"][i: i + self.batchSize]
				labels = self.db["labels"][i: i + self.batchSize]

				# If categorical_crossentropy loss is used than label binarization must be used.
				if self.binarize:
					labels = np_utils.to_categorical(labels, self.classesNum)

				processedImgs = []
				# Iterate over images in sub list
				for image in images:

					# Mean substraction
					if self.mean is not None:
						(B, G, R) = cv2.split(image.astype("float32"))
						R = R - self.mean[0]
						G = G - self.mean[1]
						B = B - self.mean[2]
						image = cv2.merge([B, G, R])

					# Image resize.
					if self.width is not None and self.height is not None:
						image = cv2.resize(image, (self.width, self.height),  interpolation=self.inter)

					processedImgs.append(img_to_array(image))
				images = np.array(processedImgs)
				images = images/128
				# Mage data augumentation
				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images,
						labels, batch_size=self.batchSize))

				# Pass to training process.
				yield (images, labels)

			epochs += 1