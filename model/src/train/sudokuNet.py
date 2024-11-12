import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils.load_data import load_data
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

import cv2


class sudokuNet:

    # Model related
    def __init__(self):
        # Load data
        self.ds_train, self.ds_test, self.ds_info = load_data()
        self.model = None
        self.file_path = 'output/'

    def get_model(self):
        return self.model

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
        print("Model loaded successfully!")

    # static methods

    @staticmethod
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    # Training related

    def train(self, epochs=5, batch_size=128):
        print("Training...")
        self.epoch = epochs

        # Create the model with Input layer instead of input_shape
        self.model = tf.keras.Sequential([


            # Input layer specifying the input shape
            layers.Input(shape=[28, 28, 1]),  # Input layer

            # First Convolutional Block (3 layers of Conv2D with 64 filters)
            layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
            layers.MaxPool2D(),

            # Classifier Head
            layers.Flatten(),
            layers.Dense(units=10, activation="softmax"),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Assuming you have the process_data function for loading data
        ds_train, ds_test = self.process_data(batch_size)

        # Define early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=epochs // 5,
            restore_best_weights=True,
        )

        # Train the model
        self.model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_test,
            callbacks=[early_stopping]
        )

        self.save_model()
        self.save_plot()


    def process_data(self, batch_size=128):

        # create an inverted dataset
        ds_train_inverted = self.ds_train.map(
            lambda image, label: (1 - image, label))
        ds_test_inverted = self.ds_test.map(
            lambda image, label: (1 - image, label))

        # combine the datasets

        self.ds_train = self.ds_train.concatenate(ds_train_inverted)
        self.ds_test = self.ds_test.concatenate(ds_test_inverted)

        # Prepare the data
        ds_train = self.ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(self.ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        ds_test = self.ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


        return ds_train, ds_test
    
    def preprocess_image(self, image_path):
        image = tf.keras.utils.load_img(image_path, target_size=(28, 28), color_mode='grayscale')  # adjust size and color mode as needed

        # invert the colors of the image
        image = cv2.bitwise_not(np.array(image))

        # Convert to array and normalize pixel values
        image = self.normalize_img(np.array(image), 0)[0]

        # Expand dimensions to match model input shape
        image = np.expand_dims(image, axis=0)

        # save the image
        file_name = 'output/processedImages/' + os.path.basename(image_path)
        plt.imsave(file_name, image[0], cmap='gray')

        return image

    # Evaluation related
    def predict(self, image_path):
        if self.model is None:
            print("No model to predict!")
            return
        
        image = self.preprocess_image(image_path)

        return np.argmax(self.model.predict(image))
    
    def evaluate(self):
        if self.model is None:
            print("No model to evaluate!")
            return
        
        ds_train, ds_test = self.process_data()
        self.model.evaluate(ds_test)

    # data helper functions 
    
    def get_summary(self):
        if self.model is None:
            print("No model to summarize!")
            return
        self.model.summary()

    def save_plot(self):
        
        # Get the training history
        history = self.get_model().history.history

        # Plot the training history 
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')

        # save the plot
        file_name = datetime.datetime.now().strftime(f"%m%d-%H%M%S-epochs{self.epoch}")
        plt.savefig('output/plot/' + file_name + '.png')
    
    def save_model(self, file_name=None):
        if self.model is None:
            return
        if file_name is None:
            file_name = datetime.datetime.now().strftime(f"%m%d-%H%M%S-epochs{self.epoch}")

        self.model.save(self.file_path + '/model/' + file_name + '.keras')
        history = self.model.history
        with open(self.file_path + '/history/' + file_name + '.pickle', 'wb') as f:
            pickle.dump(history.history, f)