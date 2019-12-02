# ai-nimals
MIT license :)<br>

To run my code you should execute file by file in below order. First you want to scrap data from google. Than you wan your crops and only images of birds. When this is done you want to clean your data - get rid of similar and duplicate data. Next step is to generate h5py file containers for training neural networks.

<b>1. ai_nimals_scrapper.py:</b><br>
This script was created for downloading multiple images from Google. Images scraping in python description is avilable on<br> https://ai-experiments.com/images-scraping-python-selenium/

<b>2. ai_nimals_prepare_scrapped.py:</b><br>
This script was created to easily filter out no bird images from downloaded data set. Data cleaning with python script description is avilable on<br>
https://ai-experiments.com/data-cleaning-python-mobilenet/

<b> 3. ai_nimals_prepare_cropped.py:</b><br>
Work in progress. This is next step of data cleaning with python script. The idea is to delete similar or duplicated images. Description avilable at <br>
https://ai-experiments.com/remove-similar-images-opencv/

<b> 4. ai_nimals_dataset_splitter.py:</b><br>
In a meantime it turned out that before I start generating h5py faile containers I need a script for splitting (copying) data to training, testing and validationg directories. Detailed description can be found below.<br>
https://ai-experiments.com/training-data-splitting-python/

<b> 5. ai_nimals_h5py_generator.py:</b><br>
Generates h5py file containers. Deatiled description:<br>
https://ai-experiments.com/hdf5-python-h5py-library/

<b> 6. ai_nimals_train_alexnet.py</b><br>
Here you can find the training script for alexnet. I also describe how to deal with killing your script by linux kernel and why it is importatnt to watch your SWAP memory size in corelation with bathsize.<br>
https://ai-experiments.com/model-training-in-keras-and-python/

<br>
<b> 7. ai_nimals_app.py</b><br>
This is main application. This application is using VGG16 network. So far I have to add some modifications and describe it on my blog.

Helpers:
<br>
<b> 1. DatasetWriter.py:</b><br>
H5PY helper class for generating containers. <br>
Usage:
writer = DatasetWriter("datasetFileName.h5py", (len(dataPaths), 256, 256, 3))
writer.add(cv2Image[i], label[i], i) - i is an iterator. You can loop over your directory
writer.close - closes your h5py file
<br><br>
<b>Note. This repository is a WIP. The plan is to create nice robust calss for downloading images of MobileNetSSD classes. And maybe YOLO also :). Firstly I want to have full functionality before I rework this code<b>
 
