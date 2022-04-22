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
num_fed_round = 5

NUMBER_CLASSES = 15
PATH = '/home/gargano/dataset/dataWithoutMasks'
USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS_TWO = ['Amparore', 'Baccega']
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
num_clients = len(USERS)
all_models = []
collaborators = []


class Aggregator(object):

    def __init__(self, model, num_clients, collaborators, num_fed_round):
        self.model = model
        self.num_clients = num_clients
        self.collaborators = collaborators
        self.num_fed_round = num_fed_round


    #Initialize model from safe_drive_SeqCNN.py
    def initialize_local_model(self):
        #return generate_simplyfied_model_safe_drive()
        return generate_model_safe_drive()


    #Take a list of models and return the mean of the models (mean of the weights)
    def local_update(self, models):

        print('Federated learning aggregation...')
        
        
        # initialize empty weights
        weights = np.array(self.model.get_weights(), dtype='object')*0  
        
        
        for client_model in models:
                client_weights = client_model.get_weights()
                weights = weights + np.array(client_weights, dtype='object')

        
        # aggregate weights, computing the mean
        FL_weights = weights/len(models)
    
        # set aggregated weights
        self.model.set_weights(FL_weights)

        return self.model


    #After we got the data as tensor we start the fake federation learning with 29 users
    def start_round_training(self, data, index_user):
    
        
        x_train, y_train, x_test, y_test = data

        #For each users in users we will do the training using the data of the user
        #for user in USERS:

            #print("\nUser data loading number" + str(USERS.index(user)))

            #x_train, x_test, y_train, y_test = start_simulated_federated_learning_loading_data(USERS.index(user))

        print('\n\nStart training model of user number ', index_user, '\n\n')

        fit_model_federation(self.collaborators[index_user].model, x_train, y_train, x_test, y_test)

        print('\n\nEnd training model of user number ', index_user , '\n\n')

        #After the training we will send the updated model to the aggregation server
        all_models.append(self.collaborators[index_user].model)


    def send_model_to_collaborators(self):

        for collaborator in self.collaborators:
            collaborator.model = self.model

#Code for collaborator class in simulated federation learning, collaboratos take the model from the aggregator that initialize it
#Data is a n-uple of (x_train, y_train, x_test, y_test)
class Collaborator(object):
    
        def __init__(self, model, data):
            self.model = model
            self.data = data
    
        #Take the model from the aggregator and train the model with the data of the user


#Describing workflow: 
# 1. Initialize the model
# 2. For each round t=1,…:
# 3. Assign the model to the collaborators, each collarator has the same initial model
# 4. Assign collaborators to the aggregation server
# 5. In the collaborator we 




#Initialize the aggregator model
model = generate_model_safe_drive()
#model = generate_simplyfied_model_safe_drive()
model = model_compile(model)


#Initialize the collaborators with own data and model
for user in USERS:
    x_train, x_test, y_train, y_test = loading_data_user(USERS.index(user))
    data = (x_train, y_train, x_test, y_test)
    collaborator = Collaborator(model, data)
    collaborators.append(collaborator)


#Initialize the collaborator 
aggregator = Aggregator(model, num_clients, collaborators, num_fed_round)


#Start the training of the model
for round in range(num_fed_round):
    print('Federated learning round: ',round+1, '\n\n')

    for i in range(len(USERS)):
        aggregator.start_round_training(aggregator.collaborators[i].data, i)

    #local update of the model in the aggregato
    aggregator.model = aggregator.local_update(all_models)
    print('\n\nSending model to collaborators...\n\n')
    aggregator.send_model_to_collaborators()

print('End of federated learning\n\nEvaluation of the model...\n\n')
validation = aggregator.collaborators[random.randint(0,len(USERS)-1)].data[2]
trained_model_evaluation(aggregator.model, validation)

#################################################



