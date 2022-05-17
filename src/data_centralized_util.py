from glob import glob
from importlib.resources import path
import random
from re import M
import os.path
from sre_constants import JUMP
import time
from scipy import rand
from tqdm import tqdm 
import numpy as np
import cv2
from cProfile import label
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

from safe_drive_SeqCNN_wOpenCV import *



def train_test_split_on_all_data(X, y, names):
    
    #Use all USERS data indices
    indices = [i for i, x in enumerate(names) if x in USERS]
    x_test = [e for i, e in enumerate(X) if i in indices]
    x_train = [e for i, e in enumerate(X) if i not in indices]
    y_test = [e for i, e in enumerate(y) if i in indices]
    y_train = [e for i, e in enumerate(y) if i not in indices]
    
    return x_train, x_test, y_train, y_test   



def load_train_all_users():
    
    start_time = time.time()
    train_images = [] 
    train_names = [] 
    train_labels = []

    for classed in tqdm(range(NUMBER_CLASSES)):
        
        print('Loading directory c{}'.format(classed))
                        
        
        #print(glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png')))  
        #files = glob(os.path.join(PATH, CATEGORIES[classed], USERS[user_chosen] + '*.png'))

        #files = glob(os.path.join(PATH, CATEGORIES[classed], USERS[current_user_index] + '*.png'))
        
        files = glob(os.path.join(PATH, CATEGORIES[classed], '*.png'))

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


def normalize_train_full_dataset(labels, names, X):
    #One shot encoding
    #y = np.utils.to_categorical(labels, NUMBER_CLASSES)
    

    #Print labels
    #print("Printing labels\n", labels, "\n\nLabels lenght: ", len(labels))
    
    y = tf.keras.utils.to_categorical(labels, NUMBER_CLASSES)
    
    x_train, x_test, y_train, y_test = train_test_split_on_all_data(X,y,names)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train = np.array(x_train, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    x_test = np.array(x_test, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    
    return x_train, x_test, y_train, y_test


def train_test_split_on_all_data(X, y, names):
    
    #Use all USERS data indices
    indices = [i for i, x in enumerate(names) if x in USERS]
    x_test = [e for i, e in enumerate(X) if i in indices]
    x_train = [e for i, e in enumerate(X) if i not in indices]
    y_test = [e for i, e in enumerate(y) if i in indices]
    y_train = [e for i, e in enumerate(y) if i not in indices]
    
    return x_train, x_test, y_train, y_test   


def normalize_train_data(labels, names, X):
    #One shot encoding
    #y = np.utils.to_categorical(labels, NUMBER_CLASSES)
    

    #Print labels
    #print("Printing labels\n", labels, "\n\nLabels lenght: ", len(labels))
    
    y = tf.keras.utils.to_categorical(labels, NUMBER_CLASSES)
    
    x_train, x_test, y_train, y_test = train_test_split_on_all_data(X,y,names)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_train = np.array(x_train, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    x_test = np.array(x_test, dtype=np.float32).reshape(-1,img_cols,img_rows,color_type)
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    
    return x_train, x_test, y_train, y_test


#Loading data for all users for centralized training   
def load_full_dataset():


    img = load_train_all_users()

    #normalize_img(img)

    #TODO: Split the dataset into train and test for each user
     
    X, labels, names = img
  
    #Can't run this because we don't have this much ram to store all the dataset
    #So we're going to pick only one user data per training
    x_train, x_test, y_train, y_test =  normalize_train_data(labels, names, X)

    #For validation, stratify is used to use all classes in the test set
    x_train, x_test, y_train, y_test = train_test_split(x_train , y_train, test_size=0.2, random_state=42, stratify=y_train)

    print("\n\nTrain test split done\n\n")
    
    return x_train, x_test, y_train, y_test 

