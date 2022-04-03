from safe_drive_SeqCNN_wOpenCV import *
from safe_drive_SeqCNN import *


from glob import glob
from importlib.resources import path
import random
import time
from tqdm import tqdm 
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import cv2
from cProfile import label
import os
import PIL
import PIL.Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
#import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt



''' 
Aggregation server

1. initialize model W0
2. for each round t=1,…:
3. Broadcast Wt−1 to all collaborators
4. select C eligible participants
5. foreach|| participant p: 
6.  wtp = LocalUpdate(p)
7. wt = aggregate(∀p, wtp)
 '''

'''
Collaborator: Local Update
get model from aggregation server
select batch (all of current user)
gb <- compute gradient for batch
send model to aggregation server
''' 

#Create two classes, one is the Aggregator and the other is the Collaborator

# Aggregator: Initialize the model, send the model to the collaborators (there are 30 collaborators), then every 
# Collaborator will update the model with the initial weights of the Collaborator, 
# finally the aggregator will extract the weights 
# of the Collaborators and update the model with the mean of the weights. This is repeated for some epochs.

# Collaborator: Do one step of SGD with the data of one user and then send the updated model to the aggregator

#Declaring global variables

color_type = 3
img_cols = 240
img_rows = 240


NUMBER_CLASSES = 15
PATH = '/home/gargano/dataset/dataWithoutMasks'
USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
MODELS = []



class Aggregator(object):

    def __init__(self, model, num_clients):
        self.model = model
        self.num_clients = num_clients



    #Initialize model from safe_drive_SeqCNN.py
    def initialize_local_model(self):
        return generate_model_safe_drive()


    #After we got the data as tensor we start the fake federation learning with 30 users
    def start_round_training(self):
        

        #For each users in users we will do the training using the data of the user
        for user in USERS:

            print("\nUser data loading number" + str(USERS.index(user)))

            start_simulated_federated_learning_loading_data(USERS.index(user))

            


    #Take a list of models and return the mean of the models (mean of the weights)
    def local_update(self, models):

        models = MODELS
        weights = []

        #Take the weights of the models and compute the mean then return the weights to an updated model
        for model in models:
            #store the weights of current model
            weights.append(model.get_weights())
    
        #compute the mean of the weights
        weights = np.mean(weights, axis=0)

        #update the model with the mean of the weights
        self.model.set_weights(weights)

        return self.model

    def extract_weights(self):
            return np.mean(self.weights, axis=0)


    def update(self, weights):
        self.weights.append(weights)
        self.losses.append(self.model.evaluate(self.weights))



#Code for fake collaborator class in simulted federation learning
class Collaborator(object):

    def __init__(self, model, num_clients):
        self.model = model
        self.num_clients = num_clients


    def update(self, weights):
        self.weights.append(weights)
        self.losses.append(self.model.evaluate(self.weights))

    def extract_weights(self):
        return self.weights[-1]





aggregator = Aggregator(generate_model_safe_drive(), 30)

aggregator.start_round_training()
