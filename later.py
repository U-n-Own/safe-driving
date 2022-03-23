
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
