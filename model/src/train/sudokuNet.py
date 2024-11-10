import tensorflow as tf
from utils.load_data import load_data
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

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

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=epochs // 5,
            restore_best_weights=True,
        )

        # Create the model
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        ds_train, ds_test = self.process_data(batch_size)

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
        # Prepare the data
        ds_train = self.ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(self.ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Prepare the test data
        ds_test = self.ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(batch_size)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        return ds_train, ds_test
    
    def preprocess_image(self, image_path):

        image = tf.keras.utils.load_img(image_path, target_size=(28, 28), color_mode='grayscale')  # adjust size and color mode as needed

        # Convert to array and normalize pixel values
        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0  # Normalize to [0, 1] range

        # Expand dimensions to match model input shape
        image = np.expand_dims(image, axis=0)

        # save the image
        file_name = 'output/processedImages/' + os.path.basename(image_path)
        plt.imsave(file_name, image[0, :, :, 0], cmap='gray')

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
