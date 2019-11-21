from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2

class DataGenerator:
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

	def setimageresizer(self, width, height, inter = cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.inter = inter

	def setmeanpreprocessor(self, r, g, b):
		self.mean = [r,g,b]

	def generator(self, passes=np.inf):
		epochs = 0

		while epochs < passes:
			for i in np.arange(0, self.numImages, self.batchSize):
				images = self.db["images"][i: i + self.batchSize]
				labels = self.db["labels"][i: i + self.batchSize]

				if self.binarize:
					labels = np_utils.to_categorical(labels, self.classesNum)

				processedImgs = []
				for image in images:
					if self.mean is not None:
						(B, G, R) = cv2.split(image.astype("float32"))
						R = R - self.mean[0]
						G = G - self.mean[1]
						B = B - self.mean[2]
						image = cv2.merge([B, G, R])

					if self.width is not None and self.height is not None:
						image = cv2.resize(image, (self.width, self.height),  interpolation=self.inter)

					processedImgs.append(img_to_array(image))

				images = np.array(processedImgs)

				if self.aug is not None:
					(images, labels) = next(self.aug.flow(images,
						labels, batch_size=self.batchSize))

				yield (images, labels)

			epochs += 1