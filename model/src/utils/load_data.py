import tensorflow_datasets as tfds

def load_data():
    print("Loading data...")
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    print("Data loaded successfully!")
    return ds_train, ds_test, ds_info

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Helper function to generate images of digits using fonts
def generate_computer_written_digits(font_paths, image_size=(28, 28), num_per_digit=1000):
    """
    Generate synthetic computer-written digit images.

    Args:
        font_paths (list): List of paths to font files.
        image_size (tuple): Size of the output images (height, width).
        num_per_digit (int): Number of images to generate per digit.

    Returns:
        images (numpy.ndarray): Array of generated images.
        labels (numpy.ndarray): Array of corresponding labels.
    """
    digits = list(range(10))
    images = []
    labels = []
    
    for digit in digits:
        for font_path in font_paths:
            font = ImageFont.truetype(font_path, size=image_size[0])
            for _ in range(num_per_digit // len(font_paths)):
                img = Image.new('L', image_size, color=0)  # White background
                draw = ImageDraw.Draw(img)

                # Center the digit in the image
                bbox = draw.textbbox((0, 0), str(digit), font=font)
                text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                
                position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
                draw.text(position, str(digit), fill=255, font=font)  # Black digit

                # Convert to numpy array and append
                images.append(np.array(img, dtype=np.uint8))
                labels.append(digit)

    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)
    return images, labels

# Create the dataset in the required format
def load_computer_written_data(image_size=(28, 28), num_per_digit=1000):
    print("Generating computer-written digits...")

    # Provide a list of font paths (ensure these fonts exist on your system)
    font_paths = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc"
    ]


    images, labels = generate_computer_written_digits(font_paths, image_size, num_per_digit)

    # Convert to TensorFlow datasets
    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((images, labels))

    # Split into train and test sets
    total_samples = len(labels)
    train_size = int(total_samples * 0.8)
    ds_train = dataset.take(train_size)
    ds_test = dataset.skip(train_size)

    print("Computer-written digit data generated successfully!")
    return ds_train, ds_test

def load_data_with_computer_written():
    # Load MNIST data
    ds_train_mnist, ds_test_mnist, ds_info = load_data()

    # Normalize and expand dimensions for MNIST
    def preprocess_mnist(image, label):
        # image = tf.expand_dims(image, axis=-1)  # Add channel dimension
        image = tf.cast(image, tf.float32) / 255.0  # Normalize
        return image, label

    ds_train_mnist = ds_train_mnist.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_mnist = ds_test_mnist.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    
    for images, labels in ds_train_mnist.take(1):
        print(f"Images shape: {images.shape}, jdioedoehode Labels shape: {labels.shape}")

    # Load and preprocess computer-generated data
    ds_train_comp, ds_test_comp = load_computer_written_data()

    def preprocess_computer_written(image, label):
        # Add channel dimension only if not already present
        if len(image.shape) != 3:  # Shape (28, 28, 1) is correct
            image = tf.expand_dims(image, axis=-1)
        image = tf.cast(image, tf.float32) / 255.0  # Normalize
        return image, label


    ds_train_comp = ds_train_comp.map(preprocess_computer_written, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_comp = ds_test_comp.map(preprocess_computer_written, num_parallel_calls=tf.data.AUTOTUNE)
    
    for images, labels in ds_train_comp.take(1):
        print(f"Images shape: {images.shape}, for datatallene Labels shape: {labels.shape}")

    # Combine MNIST and computer-written datasets
    ds_train = ds_train_mnist.concatenate(ds_train_comp)
    ds_test = ds_test_mnist.concatenate(ds_test_comp)

    # Shuffle and batch the combined datasets
    ds_train = ds_train.shuffle(buffer_size=10000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test, ds_info
