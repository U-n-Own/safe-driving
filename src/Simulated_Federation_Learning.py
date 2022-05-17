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

#Declaring global variables

color_type = 3
img_cols = 240
img_rows = 240
num_fed_round = 10

NUMBER_CLASSES = 15
PATH = '/home/gargano/dataset/dataWithoutMasks'
ALL_USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider']
USERS_TEST =['Thomas']
USERS_TWO = ['Amparore', 'Baccega']
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
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

    def plot_results_federation(self, fed_acc, fed_acc_used):

        plt.figure(figsize=(5,4))
        plt.plot(fed_acc,label='Federated Learning')
        plt.plot(fed_acc_used, label='Federated Learning data used')
        #plt.plot(history_centralized_learning.history['val_accuracy'],label='Centralised learning')
        plt.xlabel('Number of epochs')
        plt.ylabel('Validation accuracy')
        plt.legend()
        plt.grid()
        plt.xticks(np.arange(0,20,1),np.arange(1,21,1))
        plt.xlim(0,20)
        plt.savefig('plots/federated_learning_plot_after_rework-federation.png',dpi=150)

    def plot_results_centrlized(self):

        plt.figure(figsize=(5,4))
        plt.plot(history_centralized_learning.history['val_accuracy'],label='Centralised learning')
        plt.xlabel('Number of epochs')
        plt.ylabel('Validation accuracy')
        plt.legend()
        plt.grid()
        plt.xticks(np.arange(0,20,1),np.arange(1,21,1))
        plt.xlim(0,20)
        plt.savefig('plots/federated_learning_plot_after_rework-centrlized.png',dpi=150)

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


#   [Centralized learning] 

#history_centralized_learning = train_model_centralized()



#   [Federated Learning]

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


#Initialize the aggregator 
aggregator = Aggregator(model, num_clients, collaborators, num_fed_round)


#Start the training of the model
for round in range(num_fed_round):
    print('Federated learning round: ',round+1, '\n\n')
    #On all users but not the last that we use as a check same with the centralized
    for i in range(len(USERS) - 1):
        aggregator.train_collaborator(aggregator.collaborators[i].data, i)

    #local update of the model in the aggregator
    aggregator.model = aggregator.local_update(all_models)
    print('\n\nSending model to collaborators...\n\n')
    aggregator.send_model_to_collaborators()

    print('End of federated learning round\n\nEvaluation of the model...\n\n')
    
    #Each time we use validation set of a random user to predict the accuracy
#    random_pick = random.randint(0,len(USERS)-1)
#    x_test = aggregator.collaborators[random_pick].data[2]
#    y_test = aggregator.collaborators[random_pick].data[3]


    #Pick the last collaborator that we've not trained on
    X_test = aggregator.collaborators[-1].data[2]
    Y_test = aggregator.collaborators[-1].data[3]
    
    #Pick collaborator which data are used to train
    X_test_used = aggregator.collaborators[0].data[2]
    Y_test_used = aggregator.collaborators[0].data[3]
    
    fed_acc.append(aggregator.accuracy_federated_learning(X_test, Y_test))
    fed_acc_used.append(aggregator.accuracy_federated_learning(X_test_used, Y_test_used))

    
    aggregator.plot_results_federation(fed_acc, fed_acc_used)

    #Plot the results, on all users
    #TODO: try to plot for each user in a separate graph

    #aggregator.plot_results_federation(fed_acc, history_centralized_learning)
    
    #aggregator.prediction_aggregation(x_test, y_test)
    #trained_model_evaluation(aggregator.model, validation)

#################################################
