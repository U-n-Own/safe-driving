from glob import glob
from importlib.resources import path
import random
from re import M
import os.path
from sre_constants import JUMP
import time
from scipy import rand
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

#Declaring variables

color_type = 3
#Original is 640x480
img_cols = 240
img_rows = 240
NUMBER_CLASSES = 15
PATH = '/home/gargano/dataset/dataWithoutMasks'
USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
USERS_DONE = []
USER_TRAINED = list(range(len(USERS)))
root_dir = '/home/gargano/'
DATADIR = "/home/gargano/dataset/dataWithoutMasks"
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
i = 0

def get_cv2_image(path):
        # Loading as Grayscale image
        if color_type == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif color_type == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        # crop to 480x480
        #img = img[:,80:560]
        # Reduce size to 240x240
        img = cv2.resize(img, (img_rows, img_cols)) 
        return img

# Training
# Now is loading all users
#we want only single user 
def load_train_single_user(current_user_index):
    start_time = time.time()
    train_images = [] 
    train_names = [] 
    train_labels = []
   

    #user_chosen = pick_random_user()
 
    for classed in tqdm(range(NUMBER_CLASSES)):
        
        print('Loading directory c{}'.format(classed))
                         
        
        #print(glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png')))  
        #files = glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png'))

        files = glob(os.path.join(PATH, CATEGORIES[classed], USERS[current_user_index] + '*.png'))
        
        print(len(files))
        for file in files:
            img = get_cv2_image(file)
            train_images.append(img)
            train_names.append(file)
            # train_labels.append(classed)
            train_labels.append(classed // 1)
            
    print("Data Loaded in {} second".format(time.time() - start_time))

    X = train_images
    labels = train_labels
    names = train_names

    return X, labels, names, current_user_index

def normalize_img( img):
    #img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def normalize_train_data_user(user, labels, names, X):
    #One shot encoding
    #y = np.utils.to_categorical(labels, NUMBER_CLASSES)
    

    #Print labels
    #print("Printing labels\n", labels, "\n\nLabels lenght: ", len(labels))
    
    y = tf.keras.utils.to_categorical(labels, NUMBER_CLASSES)
    
    x_train, x_test, y_train, y_test = train_test_split_on_single_user(X,y,names,user)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train = np.array(x_train, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    x_test = np.array(x_test, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    
    return x_train, x_test, y_train, y_test

    

#Structure of dataset    

#Each instance work on it's own data on 29/30 users
#In federate n copy of model e
#Import dataset for training the dataset is divided in dataset/dataWithoutMasks/c00.. until c14
#There are 15 classes one for each label of action per users
#There are 30 users in the dataset with 200 image per user and each user can have 15 actions
#Path to folders is ./dataset/dataWithoutMasks/c00..c14

#How we want to train the model

#We need a function that splits the data per user, pick all data from a user and then split it in train and test for a single model
#All the data of one user is used for test and the 29/30 of the users data is used for training
#The file a are in a format like this: username_number_of_image.png
#We pick only the first 29/30 users' data for training and the last one for test
#Then we cycle and we do this for each users, so every model has one of these cycled data and it's being trained separately in only 1 epoch or 1 step of SGD
#Finally we extract the weights of each model and we compute the mean of the weights of all the models to get the final model, this is cycled for some epochs

#We use only the data of an user to train the model
#An user is picked in the dictionary USERS and then we pick only the image with the same name as the user

def train_test_split_on_single_user(X, y, names, user):
    
    #Extract the index of the user in USERS
    indices = [i for i, x in enumerate(names) if x.startswith(USERS[user])]
    x_test = [e for i, e in enumerate(X) if i in indices]
    x_train = [e for i, e in enumerate(X) if i not in indices]
    y_test = [e for i, e in enumerate(y) if i in indices]
    y_train = [e for i, e in enumerate(y) if i not in indices]
    
    return x_train, x_test, y_train, y_test  
  
#Simulation of federated learning using 30 users and using a simple iterative workflow    
def loading_data_user(current_user_index):

    print('Loading dataset for user ', current_user_index , '...\n\n')

    img = load_train_single_user(current_user_index)

    #normalize_img(img)

    #TODO: Split the dataset into train and test for each user
     
    X, labels, names, usr = img
  
    #Can't run this because we don't have this much ram to store all the dataset
    #So we're going to pick only one user data per training
    x_train, x_test, y_train, y_test =  normalize_train_data_user(usr, labels, names, X)

    #For validation, stratify is used to use all classes in the test set
    x_train, x_test, y_train, y_test = train_test_split(x_train , y_train, test_size=0.2, random_state=42, stratify=y_train)

    print("\n\nTrain test split done\n\n")
    
    return x_train, x_test, y_train, y_test 








# ----------------------------------------------------------------------------------------------------------------------

#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y
'''