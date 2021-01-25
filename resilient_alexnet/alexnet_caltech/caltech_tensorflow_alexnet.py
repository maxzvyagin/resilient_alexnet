### Tensorflow/Keras implementation of the AlexNext architectue for CIFAR100
import tensorflow as tf
from tensorflow import keras
import os
import tensorflow_datasets as tfds
import numpy as np
import pickle

def transform(i):
    return i.astype(float)

class Caltech_TensorFlow_AlexNet:
    def __init__(self, config):
        # tf.debugging.enable_check_numerics()
        tf.keras.backend.set_image_data_format('channels_first')
        ### DIFFERENT RANDOM SEED###
        tf.random.set_seed(0)
        b = int(config['batch_size'])
        # (self.x_train, self.y_train), (self.x_test, self.y_test) = get_caltech()
        # self.train, self.test = tfds.load('caltech101', split=['train', 'test'], shuffle_files=False)
        self.training_loss_history = None
        self.val_acc_history = None
        self.val_loss_history = None
        f = open('/lus/theta-fs0/projects/CVD-Mol-AI/mzvyagin/alexnet_datasets/caltech_splits.pkl', 'rb')
        data = pickle.load(f)
        (self.x_train, self.y_train), (self.x_val, self.y_val), (self.x_test, self.y_test) = data
        f.close()
        self.train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(int(config['batch_size']))
        self.val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(int(config['batch_size']))
        self.test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(int(config['batch_size']))
        # self.x_train = list(map(transform, self.x_train))
        # self.x_val = list(map(transform, self.x_val))
        # self.x_test = list(map(transform, self.x_test))
        classes = 102
        self.model = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(11, 11), strides=4, activation='relu', input_shape=(3, 256, 256),
                                kernel_initializer='he_uniform', padding="same"),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=1, activation='relu', padding="same",
                                kernel_initializer='he_uniform'),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding="same",
                                kernel_initializer='he_uniform'),
            keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=1, activation='relu', padding="same",
                                kernel_initializer='he_uniform'),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu', padding="same",
                                kernel_initializer='he_uniform'),
            keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
            keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dropout(config['dropout']),
            keras.layers.Dense(4096, activation='relu', kernel_initializer='he_uniform'),
            keras.layers.Dropout(config['dropout']),
            keras.layers.Dense(classes, activation='relu', kernel_initializer='he_uniform')
        ])

        opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'], clipvalue=0.5)
        self.model.compile(optimizer=opt,
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])
        self.config = config

    def fit(self):
        res = self.model.fit(self.train_data, epochs=self.config['epochs'],
                             batch_size=int(self.config['batch_size']), validation_data=self.val_data,
                             shuffle=False)
        self.training_loss_history = res.history['loss']
        self.val_loss_history = res.history['val_loss']
        self.val_acc_history = res.history['val_accuracy']
        return res

    def test(self):
        res_test = self.model.evaluate(self.test_data)
        # res_test = self.model.evaluate(self.test)
        return res_test[1]

def fashion_tf_objective(config):
    model = Caltech_TensorFlow_AlexNet(config)
    model.fit()
    accuracy = model.test()
    return accuracy, model.model, model.training_loss_history, model.val_loss_history, model.val_acc_history

def standardize(i):
    new = (tf.image.resize_with_crop_or_pad(i, 300, 200)/255).numpy()
    if np.all(new==0):
        print("Found an all zero")
    return new

def get_caltech():
    """ Returns test, train split of Caltech data"""
    # first try loading from cache object, otherwise load from scratch

    train, test = tfds.load('caltech101', split=['train', 'test'], shuffle_files=False)
    train = list(train)
    train_x = [standardize(pair['image']) for pair in train]
    train_y = [pair['label'].numpy().item() for pair in train]
    test = list(test)
    test_x = [standardize(pair['image']) for pair in test]
    test_y = [pair['label'].numpy().item() for pair in test]
    return (np.array(train_x), np.array(train_y)), (np.array(test_x), np.array(test_y))

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    test_config = {'batch_size': 1, 'learning_rate': .1, 'epochs': 100, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = fashion_tf_objective(test_config)