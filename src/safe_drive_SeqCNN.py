from cProfile import label
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tkinter import Y
import numpy as np
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

batch_size = 32
img_height = 180
img_width = 180

#Import dataset for training the dataset is divided in dataset/dataWithoutMasks/c00.. until c14
#There are 15 classes one for each label of action per users
#There are 30 users in the dataset with 200 image per user and each user can have 15 actions
#This function will import the dataset and divide it in train and test set

#Path to folders is ./dataset/dataWithoutMasks/c00..c14

dataset_to_train = tf.keras.preprocessing.image_dataset_from_directory(
   # '../dataset/dataWithoutMasks',  
    '/home/gargano/dataset/dataWithoutMasks',
#    labels = 'inferred',
    label_mode = "categorical", #user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(img_height, img_width), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    class_names=[c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14],
    batch_size=batch_size,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = 'training'
)

dataset_to_validate = tf.keras.preprocessing.image_dataset_from_directory(
    #'/home/gargano/safe-driving/datasets/dataWithoutMasks/',
    '/home/gargano/dataset/dataWithoutMasks',
    #labels = 'inferred',
    label_mode = "categorical", #user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(img_height, img_width), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    class_names=[c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14],
    batch_size=batch_size,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = 'validation'
)

''' 
daset_test = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/dataWithoutMasks/',
    labels = 'inferred',
    label_mode = "categorical", 
    image_size=(256, 256), 
    class_names=[c00all],
    batch_size=batch_size,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = 'test' 
) '''

#Trying to visualize the dataset
class_names = dataset_to_train.class_names
print(class_names)
#image_batch è un tensore della forma (32, 256, 256, 3) .
#Si tratta di un batch di 32 immagini di forma 256x256x3 (l'ultima dimensione si riferisce ai canali colore RGB).
for image_batch, labels_batch in dataset_to_train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
#Chiamare .numpy() sui tensori image_batch ed labels_batch per convertirli in un numpy.ndarray .


#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y

 '''






