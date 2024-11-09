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