# ai-nimals
MIT license :)

<b>1. ai_nimals_scrapper.py:</b><br>
This script was created for downloading multiple images from Google. Images scraping in python description is avilable on<br> https://ai-experiments.com/images-scraping-python-selenium/

<b>2. ai_nimals_prepare_scrapped.py:</b><br>
This script was created to easily filter out no bird images from downloaded data set. Data cleaning with python script description is avilable on<br>
https://ai-experiments.com/data-cleaning-python-mobilenet/

<b> 3. ai_nimals_prepare_cropped.py:</b><br>
Work in progress. This is next step of data cleaning with python script. The idea is to delete similar or duplicated images. Description avilable at <br>
https://ai-experiments.com/remove-similar-images-opencv/

<b> 4. ai_nimals_dataset_splitter.py:</b><br>
In a meantime it turned out that it will be more clear if I separate data splitting as new file. Before I start generating h5py faile containers of training, testing and validating data. I like those data stores as CSV files with image paths. Detailed description at:<br>
https://ai-experiments.com/training-data-splitting-python/



<br>It can be weird. But working withc AI is not only training your model. It is mostly preparing data for training. What usually takes about 80% of yor development time. I have been commiting my work for about 4 weeks. And still it's not finished. I have my scrapper, cleaner (x2), and CSV data sets splitter. It all leads to take care of h5py file containers. That is the starting point for our CNN Wonderland journey! I cant wait to show you how deep (learning) the rabbit hole goes!

<b>Note. This repository is a WIP. The plan is to create nice robust calss for downloading images of MobileNetSSD classes. And maybe YOLO also :). Firstly I want to have full functionality before I rework this code<b>
 
