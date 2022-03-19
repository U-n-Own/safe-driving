from cProfile import label
from tkinter import Y
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
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

dataset_to_train = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/dataWithoutMasks/', 
    labels = 'inferred',
    label_mode = "categorical", #user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(256, 256), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    class_names=[c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14],
    batch_size=32,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = 'training'
)

dataset_to_validate = tf.keras.preprocessing.image_dataset_from_directory(
    'datasets/dataWithoutMasks/',
    labels = 'inferred',
    label_mode = "categorical", #user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(256, 256), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    class_names=[c00,c01,c02,c03,c04,c05,c06,c07,c08,c09,c10,c11,c12,c13,c14],
    batch_size=32,
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
    batch_size=32,
    #color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 42,
    validation_split = 0.2,
    subset = 'test' 
) '''

#Trying to visualize the dataset
class_names = dataset_to_train.class_names
print(class_names)

#Show first 15 image of training set
plt.figure(figsize=(10, 10))
for images, labels in  dataset_to_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")



#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y

 '''






