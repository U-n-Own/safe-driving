
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


#In simulated fed learning 

'''
#Define a model in the same way it's defined int safe_drive_SeqCNN.py but with a class
class CNNModel(epochs = 10):
    
    # Model for image classification on 15 classes, 
    # classes consists in actions one of them is safe driving the other are action that distract the user
    # We use a CNN with 3 convolutional layers and a fully connected layer, and we use a softmax activation function for the last layer.
    def generate_model_safe_drive():

        def __init__(self, name, age):
                self.model = model
                

        model = tf.keras.Sequential([
            #Rescaling the input image to a fixed size
            tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

            #Flatten the input to a 1-D vector
            #tf.keras.layers.Flatten(input_shape=(256, 256, 3)),

            #First convolutional layer with 32 filters and a kernel size of 3x3
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),

            #Second convolutional layer with 64 filters and a kernel size of 3x3
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            #Third convolutional layer with 128 filters and a kernel size of 3x3
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

            #Flatten the output of the previous layer
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dropout(.5, input_shape=(img_height,img_width,3)),

            #Anothet fully connected layer with 512 units
            tf.keras.layers.Dense(240, activation='relu'),

            #Final layer with 15 classes
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.summary()

        return model

    #Try experimenting with different optimizers and different optimizer configs
    def model_compile(model):

        model.compile(optimizer = 'adam', 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return model

    def fit_model(model):
        model.fit(dataset_to_train, validation_data = dataset_to_validate, epochs=10)


    def trained_model_evaluation(model):
        test_loss, test_acc = model.evaluate(dataset_to_validate)
        print('\nTest accuracy:', test_acc)

    def get_weights(model):
        return model.get_weights()

    def start_training():
        model = generate_model_safe_drive()
        print("\n\n\nModel generated with success!\n\n\n")
        model = model_compile(model)
        print("\n\n\nModel compiled with success!\n\n\n")
        fit_model(model)
        print("\n\n\nModel trained with success!\n\n\n")
        #history.results()
        trained_model_evaluation(model)

'''


''' 
    #Broken function
    def plots_result_federation_clients(self, history_clients):
        
        plt.figure(figsize=(5,4))
        
        for i in range(len(USERS)-1):
            plt.plot(history_clients[i].history['val_accuracy'],label='client learning, client '+str(i+1))

        plt.xlabel('Number of epochs')
        plt.ylabel('Validation accuracy')
        plt.legend()
        plt.grid()
        plt.xticks(np.arange(0,20,1),np.arange(1,21,1))
        plt.xlim(0,20)
        plt.savefig('plots/federated_learning_plot_each_user.png',dpi=150) '''