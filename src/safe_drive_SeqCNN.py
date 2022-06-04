from cProfile import label
from data_centralized_util import *
from util_fl import *

from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf #Ver. 2.7.0
import pathlib
import matplotlib.pyplot as plt


#weight and Biases initialization
import wandb
from wandb.keras import WandbCallback

batch_size = 32
img_height = 240
img_width = 240
num_classes = 15

# Model for image classification on 15 classes, 
# classes consists in actions one of them is safe driving the other are action that distract the user
# We use a CNN with 3 convolutional layers and a fully connected layer, and we use a softmax activation function for the last layer.
def generate_model_safe_drive():

    model = tf.keras.Sequential([


        #Rescaling the input image to a fixed size
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        #Flatten the input to a 1-D vector
        #tf.keras.layers.Flatten(input_shape=(256, 256, 3)),

        #First convolutional layer with 32 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Second convolutional layer with 64 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Third convolutional layer with 128 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

        #Flatten the output of the previous layer
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

        #Anothet fully connected layer with 512 units
        tf.keras.layers.Dense(240, activation='relu'),

        #Final layer with 15 classes
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    return model

def generate_simplyfied_model_safe_drive():

    model = tf.keras.Sequential([


        #Rescaling the input image to a fixed size
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

        #First convolutional layer with 32 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Second convolutional layer with 64 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),


        tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

        #Flatten the output of the previous layer
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

        #Anothet fully connected layer with 512 units
        tf.keras.layers.Dense(240, activation='relu'),

        #Final layer with 15 classes
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    return model


#Try experimenting with different optimizers and different optimizer configs
def model_compile(model):

    model.compile(optimizer = 'adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model

def fit_model_centralized(model, x_train, x_test, y_train, y_test):

    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=10)
    
    return history

def fit_model_federation(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=1, callbacks=[WandbCallback()])
    return history


#No Federated learning, data is not distributed
def train_model_centralized():

    #dataset_to_train, dataset_to_validate = loading_dataset()
    x_train, x_test, y_train, y_test = load_full_dataset() 
    print("\n\n\t\tClassical training\n\n")
    model = generate_model_safe_drive()
    print("\n\n\nModel generated with success!\n\n\n")
    model = model_compile(model)
    print("\n\n\nModel compiled with success!\n\n\n")

    #Old fit using dataset_to_train and dataset_to_validate
    #history = old_fit_model(model, dataset_to_train, dataset_to_validate)

    #New fit using x_train and y_train
    history = fit_model_centralized(model, x_train, x_test, y_train, y_test)

    print("\n\n\nModel trained with success!\n\n\n")
    #history.results()
    #classical_accuracy = trained_model_evaluation(model, dataset_to_validate)

    return history
