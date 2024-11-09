from tensorflow import keras


def main():
    print("Hello World!")
    print("Loading MNIST dataset...")
    keras.datasets.mnist.load_data()
    print(keras.__version__)


if __name__ == "__main__":
    main()
