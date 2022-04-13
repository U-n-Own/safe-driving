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
USERS_FULL =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS =['Amparore', 'Baccega']
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
all_models = list(range(0,2))
collaborators = []
num_clients = len(USERS)

class Aggregator(object):

    def __init__(self, model, num_clients, collaborators):
        self.model = model
        self.num_clients = num_clients
        self.collaborators = collaborators
        



    #Initialize model from safe_drive_SeqCNN.py
    def initialize_local_model(self):
        #return generate_simplyfied_model_safe_drive()
        return generate_model_safe_drive()


    #Take a list of models and return the mean of the models (mean of the weights)
    def local_update(self, models):

        #models = MODELS
        weights = []
        count_layer = 0
        #Take the weights of the models and compute the mean then return the weights to an updated model

        for model in models:
            for layer in model.layers:
                #print(model.layers[0].weights)
                #print(layer.name, layer)
                weights.append(model.get_layer(layer.name).weights)
    

        #weights list contains the weights of each layer of each model
        #weights[0] contains the weights of the first layer of the first model
        #Transform weights in a list of weights for each layer of each model so that the weights are the mean of each model
        mean_of_weights = []

        for weight in weights[0]:
            mean_of_weights.append(np.mean(np.array([model.get_layer(weight.name).get_weights() for model in models]), axis=0))
    


        #weights = np.mean(weights, axis=0)

        print("\n\nCurrent shape of mean of weights, after mean\n\n")
        print(mean_of_weights.shape) # result tensor [20,]
    

        #update the model with the mean of the weights
        self.model.set_weights(mean_of_weights)

        return self.model


    #After we got the data as tensor we start the fake federation learning with 29 users
    def start_round_training(self):
        

        #For each users in users we will do the training using the data of the user
        for user in USERS:

            print("\nUser data loading number" + str(USERS.index(user)))

            x_train, x_test, y_train, y_test = start_simulated_federated_learning_loading_data(USERS.index(user))

            print("\n\nStart training model of user number" + str(USERS.index(user)))

            #Use the Collaborator model
            #fit_model_federation(, x_train, y_train, x_test, y_test)

            fit_model_federation(self.collaborators[USERS.index(user)].model, x_train, y_train, x_test, y_test)

            print("\n\nEnd training model of user number" + str(USERS.index(user)))

            #After the training we will send the updated model to the aggregation server
            # For simulating this will save the models in an np array
            all_models[USERS.index(user)] = self.collaborators[USERS.index(user)].model
            
        

            

            
            #print("Showing shape of training data")
            #print(x_train.shape) #-> (Tensor 175 240 240 3) Bath is 0.8 of the total data of this user


            #Send the model to the collaborators, we are in a simulated environment so we train with the aggregator itself
            


#Code for collaborator class in simulated federation learning, collaboratos take the model from the aggregator that initialize it
class Collaborator(object):
    
        def __init__(self, model):
            self.model = model
    
        #Take the model from the aggregator and train the model with the data of the user
        def local_update_collaborator(self, model):
            self.model = model
            
            #Train the model with the data of the user



            return self.model


#Describing workflow: 
# 1. Initialize the model
# 2. For each round t=1,…:
# 3. Assign the model to the collaborators, each collarator has the same initial model
# 4. Assign collaborators to the aggregation server
# 5. In the collaborator we 



#Initialize the aggregator
model = generate_model_safe_drive()
#model = generate_simplyfied_model_safe_drive()
model = model_compile(model)


#Initialize the collaborators
for i in range(num_clients):
    collaborator = Collaborator(model)
    collaborators.append(collaborator)

aggregator = Aggregator(model, num_clients, collaborators)


aggregator.start_round_training()

#local update of the model in the aggregator
aggregator.model = aggregator.local_update(all_models)




#################################################




#Old solution for weights computing mean

"""         # Be careful :  Maybe this is not generic and  works only for two models

        #weights is a list containing the weights of each layer per each model
        # Mean in computed in this way the first 12 layers, contains the weights of the first model
        # The mean woulde be the sum for each component in weights at the place mod12
        # So that we sum 0 and 11, 1 and 12 elements in weights and so on. Then we divde by len(models)
        
        for weight in weights:
    
                weighted_sum = weights[first_index] + weights[second_index]

                first_index=+1
                second_index+=1

        mean_weight = weighted_sum/len(models)

        print("\n\nCurrent shape of weighted_sum, after mean\n\n")
        print(mean_weight.shape)
 """
