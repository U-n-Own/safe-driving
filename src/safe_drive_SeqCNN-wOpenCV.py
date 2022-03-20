from importlib.resources import path
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import cv2
from cProfile import label
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

''' #Model for CNN image classification on 15 classes
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
 '''


#Import dataset for training the dataset is divided in dataset/dataWithoutMasks/c00.. until c14
#There are 15 classes one for each label of action per users
#There are 30 users in the dataset with 200 image per user and each user can have 15 actions
#This function will import the dataset and divide it in train and test set

#Path to folders is ./dataset/dataWithoutMasks/c00..c14

#Import dataset using OpenCV
DATADIR = "/home/gargano/dataset/dataWithoutMasks"
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category) #Path to the folder divided in 15 classes
    for img in os.listdir(path)
    img_to_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    
#Original is 640x480
IMAGE_SIZE = 256
#IMAGE_SIZE = 512
new_array = cv2.resize(img_to_array, (IMAGE_SIZE, IMAGE_SIZE))

plt.imshow(new_array)
plt.show()

#image = cv2.imread("/home/gargano/dataset/dataWithoutMasks/c00/IMAGE_NAME_HERE.png", 0) 
#cv2.imshow("image", image) #Can't show image in ssh session

''' 
daset_test = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/dataWithoutMasks/',
    labels = 'inferred',
    label_mode = "categorical", 
    image_size=(256, 256), 
    class_names=[c00all],
    batch_size=32,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = 'test' 
) '''


#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y
'''