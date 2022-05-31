from safe_drive_SeqCNN_wOpenCV import loading_data_user
from safe_drive_SeqCNN import * 
from util_fl import *

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
from tensorflow import keras
from sklearn.model_selection import train_test_split
#import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import wandb

#weight and Biases initialization
from wandb.keras import WandbCallback

wandb.init(project="safe-driving", entity="vincent") 
wandb.config = {
            "learning_rate" : 0.001,
            "batch_size" : 32, 
            "epochs" : 1,
            "loss_function" : "categorical_crossentropy"
            }

#Declaring global variables
color_type = 3
img_cols = 240
img_rows = 240
num_fed_round = 50
num_clients = len(USERS)
all_models = []
collaborators = []
fed_acc = []
fed_acc_used = []
history_clients = []

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

    #Perfome single epoch fit for each collaborator and store current epoch model
    def train_collaborator(self, data, index_user):
    
        #Problem: we can get access to data from self.collaborators[index_user].data
        #But i keep this format for how the fit_model_federation takes arguments
        x_train, y_train, x_test, y_test = data

        #print('\n\nStart training model of user number ', index_user, '\n\n')

        history_clients.append(fit_model_federation(self.collaborators[index_user].model, x_train, y_train, x_test, y_test))

        #print('\n\nEnd training model of user number ', index_user , '\n\n')

        #After the training we will send the updated model to the aggregation server
        all_models.append(self.collaborators[index_user].model)


    def send_model_to_collaborators(self):

        for collaborator in self.collaborators:
            collaborator.model = self.model

    def accuracy_federated_learning(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print('\nTest accuracy:', test_acc)
        print('\nTest loss:', test_loss)
        return test_acc

    def save_global_model(self):
        model = self.model
        print('\n\nSaving global model...\n\n')
        model.save('/home/gargano/safe-driving/src/models/model_federated_learning_' + str(time.time()) + '_exluded_' + USERS_EXCLUDED[0]  +'.h5')
        #model.save("Fed_model_" + str(time.time()) + '_exluded_' + USERS_EXCLUDED[0])

    def plot_results_federation(self, fed_acc, fed_acc_used):

        plt.figure(figsize=(5,4))
        plt.plot(fed_acc,label='Federated Learning')
        plt.plot(fed_acc_used, label='Federated Learning exluding user')
        #plt.plot(history_centralized_learning.history['val_accuracy'],label='Centralised learning')
        plt.xlabel('Number of epochs')
        plt.ylabel('Validation accuracy')
        plt.legend()
        plt.grid()
        plt.xticks(np.arange(0,20,1),np.arange(1,21,1))
        plt.xlim(0,50)
        plt.savefig('plots/federated_learning_plot_'+ USERS_EXCLUDED[0] +'_excluded.png',dpi=150)


#Code for collaborator class in simulated federation learning, collaborators take the model from the aggregator that initialize it
#Collaborator: Do one step of SGD with the data of one user and then send the updated model to the aggregator
class Collaborator(object):
    
        def __init__(self, model, data):
            self.model = model
            self.data = data
    

#Describing workflow: 
# 1. Initialize the model
# 2. For each round t=1,…:
# 3. Assign the model to the collaborators, each collarator has the same initial model
# 4. Each collaborator trains for an epoch
# 5. Assign collaborators' models weights to the old model of aggregation server 
# 6. Compute mean
# 7. Send the new model to the collaborators


#   [Federated Learning]    #

#Initialize the aggregator model
model = generate_model_safe_drive()
#model = generate_simplyfied_model_safe_drive()
model = model_compile(model)


#Initialize the collaborators with own data and model
''' for user in USERS:
    x_train, x_test, y_train, y_test = loading_data_user(USERS.index(user))
    data = (x_train, y_train, x_test, y_test)
    collaborator = Collaborator(model, data)
    collaborators.append(collaborator) '''

#Pick the collaborator that we've not trained on in this experiment
print("\n\nLoading test user data:\n\n\n")
x_train, X_test_not_used, y_train, Y_test_not_used = loading_data_user(USERS.index(USERS_EXCLUDED[0]))


for user in USERS_TRAINING:
    x_train, x_test, y_train, y_test = loading_data_user(USERS.index(user))
    data = (x_train, y_train, x_test, y_test)
    collaborator = Collaborator(model, data)
    collaborators.append(collaborator)


#Initialize the aggregator 
aggregator = Aggregator(model, num_clients, collaborators, num_fed_round)


#Start the training of the model
for round in range(num_fed_round):
    print('Federated learning round: ',round+1, '\n\n')


    for i in range(len(USERS_TRAINING)):
        aggregator.train_collaborator(aggregator.collaborators[i].data, i)


    #local update of the model in the aggregator
    aggregator.model = aggregator.local_update(all_models)
    print('\n\nSending model to collaborators...\n\n')
    aggregator.send_model_to_collaborators()

    print('End of federated learning round\n\nEvaluation of the model...\n\n')
    
    #Pick collaborator which data are used to train
    X_test_used = aggregator.collaborators[7].data[2]
    Y_test_used = aggregator.collaborators[7].data[3]
    
    fed_acc.append(aggregator.accuracy_federated_learning(X_test_not_used, Y_test_not_used))
    fed_acc_used.append(aggregator.accuracy_federated_learning(X_test_used, Y_test_used))

    
    aggregator.plot_results_federation(fed_acc, fed_acc_used)

aggregator.save_global_model()

#################################################
