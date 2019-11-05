import h5py

# Usage:
# writer = DatasetWriter("datasetFileName.h5py", (len(dataPaths), 256, 256, 3))
# writer.add(cv2Image[i], label[i], i) - i is an iterator. You can loop over your directory
# writer.close - closes your h5py file

# Helper class for H5PY data writer management.
class DatasetWriter:
    # Constuctor - opens and creates fie container.
    # @param containerPath - path where h5py file container will be saved
    # @param dimensions - dimensions of data set. Usually (len(dataPaths), 256, 256, 3) where "256, 256, 3" are image dimensions
    # @param labelsName - name of dataset column in container.
    # @param dataSetName - name of images data set column in container
    def __init__(self, containerPath, dimensions, labelsName="labels", dataSetName="images"):
        self.container = h5py.File(containerPath, "w")
        self.dataset = self.container.create_dataset(dataSetName, dimensions, dtype="float")
        self.labels = self.container.create_dataset(labelsName, (dimensions[0],), dtype="int")

    # Method adds data to file container.
    # @param image - image usualy opened by opencv
    # @param label - image's label. Typically encoded by sklearn - LabelEncoder
    # @param index - current file index
    def add(self, image, label, index):
        self.dataset[index] = image
        self.labels[index] = label

    # Method closes file container.
    def close(self):
        self.container.close()