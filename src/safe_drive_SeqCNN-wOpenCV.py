from importlib.resources import path
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import cv2
from cProfile import label
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

''' #Model for CNN image classification on 15 classes
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
 '''
 #Remove self 

def get_cv2_image(self,path):
        # Loading as Grayscale image
        if self.color_type == 1:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif self.color_type == 3:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        # crop to 480x480
        img = img[:,80:560]
        # Reduce size to 240x240
        img = cv2.resize(img, (self.img_rows, self.img_cols)) 
        return img

    def normalize_img(self, img):
        #img = tf.cast(img, dtype=tf.float32)
        # Map values in the range [-1, 1]
        return (img / 127.5) - 1.0
    
    # Training
    #Only single user 
    def load_train(self):
        start_time = time.time()
        train_images = [] 
        train_names = [] 
        train_labels = []
        # Loop over th
        for classed in tqdm(range(self.NUMBER_CLASSES)):
        # for classed in [0, 1, 2, 3, 4]:
        print('Loading directory c{}'.format(classed))
        # print(os.path.join(self.PATH, 'c' + ('0'+str(classed) if classed < 10 else str(classed)), '*.png'))
        print(os.path.join(self.PATH.format(classed), '*.png'))
        # files = glob(os.path.join(self.PATH, 'c' + ('0'+str(classed) if classed < 10 else str(classed)), '*.png'))
        files = glob(os.path.join(self.PATH.format(classed), '*.png'))
        print(len(files))
        for file in files:
            img = self.get_cv2_image(file)
            train_images.append(img)
            train_names.append(file)
            # train_labels.append(classed)
            train_labels.append(classed // 1)
                
        print("Data Loaded in {} second".format(time.time() - start_time))
        self.X = train_images
        self.labels = train_labels
        self.names = train_names
#1 user per test. Loading a single user
  def train_test_split_users(self,X,y,names,user):
        indices = [i for i, x in enumerate(names) if self.USERS[user] in x]
        x_test = [e for i, e in enumerate(X) if i in indices]
        x_train = [e for i, e in enumerate(X) if i not in indices]
        y_test = [e for i, e in enumerate(y) if i in indices]
        y_train = [e for i, e in enumerate(y) if i not in indices]
        return x_train, x_test, y_train, y_test
        
    
    def normalize_train_data_user(self,user):
        #One shot encoding
        y = np_utils.to_categorical(self.labels, self.NUMBER_CLASSES)
        x_train, x_test, y_train, y_test = self.train_test_split_users(self.X,y,self.names,user)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        x_train = np.array(x_train, dtype=np.float32).reshape(-1,self.img_cols,self.img_rows,self.color_type)
        x_test = np.array(x_test, dtype=np.float32).reshape(-1,self.img_cols,self.img_rows,self.color_type)
        # x_train = x_train / 255.0
        # x_test = x_test / 255.0
        
        return x_train, x_test, y_train, y_test

#For validation, stratify is used to use all classes in the test set
train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

USERS =['Amparore', 'Baccega', 'Basile', 'Beccuti', 'Botta', 'Castagno', 'Davide', 'DiCaro', 'DiNardo','Esposito','Francesca','Giovanni','Gunetti','Idilio','Ines','Malangone','Maurizio','Michael','MirkoLai','MirkoPolato','Olivelli','Pozzato','Riccardo','Rossana','Ruggero','Sapino','Simone','Susanna','Theseider','Thomas']
#Each instance work on it's own data on 29/30 users
#In federate n copy of model e
#Import dataset for training the dataset is divided in dataset/dataWithoutMasks/c00.. until c14
#There are 15 classes one for each label of action per users
#There are 30 users in the dataset with 200 image per user and each user can have 15 actions

#Path to folders is ./dataset/dataWithoutMasks/c00..c14

#Import dataset using OpenCV
root_dir = '/home/gargano/'
DATADIR = "/home/gargano/dataset/dataWithoutMasks"
CATEGORIES = ["c00","c01","c02","c03","c04","c05","c06","c07","c08","c09","c10","c11","c12","c13","c14"]
i = 0

for category in CATEGORIES:
    print(category + "is now being processed")
    path = os.path.join(DATADIR,category) #Path to the folder divided in 15 classes
    for img in os.listdir(path):   
        img_to_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
    
#Original is 640x480
IMAGE_SIZE = 240

#IMAGE_SIZE = 512
#new_array = cv2.resize(img_to_array, (IMAGE_SIZE, IMAGE_SIZE))


print("end of loading")
#image = cv2.imread("/home/gargano/dataset/dataWithoutMasks/c00/IMAGE_NAME_HERE.png", 0) 



#Todo: Copy the train and test set in new directory
#Data augmentations maybe
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y
'''
