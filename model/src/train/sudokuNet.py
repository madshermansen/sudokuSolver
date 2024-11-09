import tensorflow as tf
from utils.load_data import load_data
import datetime

class sudokuNet:

    @staticmethod
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def __init__(self):
        # Load data
        self.ds_train, self.ds_test, self.ds_info = load_data()
        self.model = None

    def train(self):
        print("Training...")

        # Create the model
        self.model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        # Prepare the data
        ds_train = self.ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(self.ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(128)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        # Prepare the test data
        ds_test = self.ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.batch(128)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        # Train the model
        self.model.fit(
            ds_train,
            epochs=100,
            validation_data=ds_test,
        )

        self.save_model(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        print("Training completed successfully!")

    def load_model(self, file_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(file_path)
        print("Model loaded successfully!")

    def save_model(self, file_name):
        if self.model is None:
            print("No model to save!")
            return
        print("Saving model...")
        self.model.save('output/' + file_name + '.keras', save_format='tf')
        print("Model saved successfully!")