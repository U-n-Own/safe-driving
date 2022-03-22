from cProfile import label
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf #Ver. 2.7.0
import pathlib
import matplotlib.pyplot as plt

batch_size = 32
img_height = 180
img_width = 180
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
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Second convolutional layer with 64 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Third convolutional layer with 128 filters and a kernel size of 3x3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        #Flatten the output of the previous layer
        tf.keras.layers.Flatten(),

        #Anothet fully connected layer with 512 units
        tf.keras.layers.Dense(512, activation='relu'),

        #Final layer with 15 classes
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    return model

#Try experimenting with different optimizers and different optimizer configs
def model_compile(model):

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def fit_model(model):
    model.fit(dataset_to_train, validation_data = dataset_to_validate, epochs=10)

def trained_model_evaluation(model):
    test_loss, test_acc = model.evaluate(dataset_to_validate)
    print('\nTest accuracy:', test_acc)

#Main function
def main():
    model = generate_model_safe_drive()
    print("\n\n\nModel generated with success!\n\n\n")
    model = model_compile(model)
    print("\n\n\nModel compiled with success!\n\n\n")
    fit_model(model)
    print("\n\n\nModel trained with success!\n\n\n")
    #history.results()
    trained_model_evaluation(model)


#This preprocessing does reshaping and splitting of the dataset

dataset_to_train = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/gargano/dataset/dataWithoutMasks',
    labels = 'inferred',
    label_mode = "categorical", #Maybe int? user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(img_height, img_width), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    batch_size=batch_size,
    color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = 'training'
)

dataset_to_validate = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/gargano/dataset/dataWithoutMasks',
    labels = 'inferred',
    label_mode = "categorical", #user distracted with 15 different actions or not one of the label is user not distracted , we chose categorical for one hot encoding
    image_size=(img_height, img_width), #Our is 640x480, we resize to 256x256, we can try to keep the original size. @Brief Reshape in not in this size
    batch_size=batch_size,
    color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 123,
    validation_split = 0.2,
    subset = 'validation'
)


#Saving class names
class_names = dataset_to_train.class_names
print("Visualizing class names")
print(class_names)
print("\n###################################################\n")

#Now we run the main function
main()

#--------------------------------------End-------------------------------------------------

#Todo later: 

#Data augmentations
''' 
def augumentation_imgs 
    image = tf.image.random_brightness(image, max_delta=0.07)
    return image, y

 '''


''' def generate_model():

    num_classes = len(class_names)

    model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])
 '''

 #Test set not working
''' dataset_test = tf.keras.preprocessing.image_dataset_from_directory(
    '/home/gargano/dataset/dataWithoutMasks',
    labels = 'inferred',
    label_mode = "categorical", 
    image_size=(255, 256), 
    batch_size=batch_size,
    color_mode="rgb", #Don't know what format images are can try both?
    shuffle = True,
    seed = 122,
    validation_split = -1.2,
    subset = 'test' 
)  '''

#@Info: image_batch è un tensore della forma (32, 256, 256, 3) .
#@Info: Si tratta di un batch di 32 immagini di forma 256x256x3 (l'ultima dimensione si riferisce ai canali colore RGB).
''' print("\nEnd of import dataset\n")
print("\n#############################\n")
print("Visualize dataset tensor")
for image_batch, labels_batch in dataset_to_train:
    print(image_batch.shape)
    print(labels_batch.shape)
    break '''

def results():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()