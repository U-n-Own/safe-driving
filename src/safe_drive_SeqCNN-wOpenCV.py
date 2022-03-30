from glob import glob
from importlib.resources import path
import random
from re import M
import os.path
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


def pick_random_user():

    #Maybe this is useless and we can pick one user witch a cycle

    #Pick only one user data from the users using random pick
    #To do this we pick one in USERS lenght and then we pick the image with the same name as the user in the array

    rand_user_pick = random.randint(0, len(USERS)-1)
    
    #print("User picked: {}".format(USERS[rand_user_pick]))

    if len(USERS_DONE) == len(USERS):
        print("All users done")
        return None

    if USERS[rand_user_pick] in USERS_DONE:
        return pick_random_user()

    if USERS[rand_user_pick] not in USERS_DONE:
        USERS_DONE.append(USERS[rand_user_pick])
        return rand_user_pick

# Training
# Now is loading all users
#we want only single user 
def load_train_single_user():
    start_time = time.time()
    train_images = [] 
    train_names = [] 
    train_labels = []
  

    user_chosen = pick_random_user()

    #Pick only image of user_chosen   
    
    for classed in tqdm(range(NUMBER_CLASSES)):
        
        print('Loading directory c{}'.format(classed))
                         
        
        #print(glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png')))  
        files = glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png'))

        
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

    return X, labels, names

def normalize_img( img):
    #img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

def normalize_train_data_user(user, labels, names, X):
    #One shot encoding
    #y = np.utils.to_categorical(labels, NUMBER_CLASSES)
    y = tf.keras.utils.to_categorical(labels, NUMBER_CLASSES)
    x_train, x_test, y_train, y_test = train_test_split_single_user(X,y,names,user)
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

#Structure of training

#We need a function that splits the data per user, pick all data from a user and then split it in train and test for a single model
#All the data of one user is used for test and the 29/30 of the users data is used for training
#The file a are in a format like this: username_number_of_image.png
#We pick only the first 29/30 users' data for training and the last one for test
#Then we cycle and we do this for each users, so every model has one of these cycled data and it's being trained separately in only 1 epoch or 1 step of SGD
#Finally we extract the weights of each model and we compute the mean of the weights of all the models to get the final model, this is cycled for some epochs

#We use only the data of an user to train the model
#An user is picked in the dictionary USERS and then we pick only the image with the same name as the user
def train_test_split_on_single_user(X, y, names, user):

    return True
    #Todo
    #indices = [i for i, x in enumerate(names) if USERS[user] in x]





#Where K is number of the client and w is the weight matrix of his own model
def fake_client_update(k, w):
    print("Client {} is updating".format(k))


def train_test_split_single_user():
    
    return True


#These should be classes maybe

def aggregator_update(w):
    print("Aggregator is updating")

def collaborator_update(w):
    print("Collaborator is updating")   
    

#Simulation of federated learning using 30 users and using a simple iterative workflow    
def start_fake_federated_learning():

    print("Starting federated learning simulation\n\n")

    print("Loading dataset for one user...\n\n")
    img = load_train_single_user()

    #normalize_img(img)

    #TODO: Split the dataset into train and test for each user

    labels, names, X = img
    print("Labels, names, X loaded\n\n")
    #Print one for each label, names and X
    for lab in range(1):
        print("Label {}: {}".format(lab, labels.count(lab)))


'''     print(labels)
    print("\n\n")
    print(names)
    print("\n\n")
    print(X) '''

    #Can't run this because we don't have this much ram to store all the dataset
    #So we're going to pick only one user data per training
    x_train, x_test, y_train, y_test =  normalize_train_data_user(USERS, labels, names, X)

    #For validation, stratify is used to use all classes in the test set
    tensor_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    print("Tensor shape: {}".format(tensor_test[0].shape + "\n\n\n"))
    tf.shape(tensor_test)



start_fake_federated_learning()






# ----------------------------------------------------------------------------------------------------------------------

#Todo: Copy the train and test set in new directory



#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y
'''

''' #Import dataset using OpenCV
def import_dataset_with_opencv():
    
    for category in CATEGORIES:
        print(category + "is now being processed")
        path = os.path.join(DATADIR,category) #Path to the folder divided in 15 classes
        for img in os.listdir(path):   
            img_to_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

 '''
# This may be useless to me
#1 user per test. Loading a single user
''' def train_test_split_users(X,y,names,user):
    indices = [i for i, x in enumerate(names) if USERS[user] in x]
    x_test = [e for i, e in enumerate(X) if i in indices]
    x_train = [e for i, e in enumerate(X) if i not in indices]
    y_test = [e for i, e in enumerate(y) if i in indices]
    y_train = [e for i, e in enumerate(y) if i not in indices]
    return x_train, x_test, y_train, y_test  
 '''

''' def old_split_single_usr():
     #For each user
    NUMBER_USERS = len(USERS)

    for user in range(NUMBER_USERS):
        #For each class
        for classed in range(NUMBER_CLASSES):
            #For each image
            for img in range(NUMBER_IMAGES):
                #Get the image
                img_to_array = cv2.imread(os.path.join(PATH.format(classed), img), cv2.IMREAD_COLOR)
                #Resize the image
                new_array = cv2.resize(img_to_array, img_cols, img_rows)
                #Add the image to the dataset
                dataset.append([new_array, classed])
                
    #Shuffle the dataset
    random.shuffle(dataset)
    #Split the dataset into train and test
    train = dataset[:TRAIN_SIZE]
    test = dataset[TRAIN_SIZE:]
    
    #Create the train and test sets
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for features, label in train:
        X_train.append(features)
        y_train.append(label)
        
    for features, label in test:
        X_test.append(features)
        y_test.append(label)
        
    #Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    X_test = np.array(X_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    #One-hot encode the labels
    y_train = np.utils.to_categorical(y_train, NUMBER_CLASSES)
    y_test = np.utils.to_categorical(y_test, NUMBER_CLASSES)
    
    return X_train, X_test, y_train, y_test    
     '''