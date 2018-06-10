from __future__ import print_function

import os
import sys
import numpy as np

import keras
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import backend as K

import os.path
my_path = os.path.abspath(os.path.dirname(__file__))

def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)

def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def train_model():

    batch_size = 128
    epochs = 50

    model_name = 'ssss'

    # number of filters to use
    nb_filters = 64
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)
    # number of classes
    nb_classes = 360

    X_train = np.load(os.path.join(my_path, '../data/processed/training/training_X.npy'))
    Y_train = np.load(os.path.join(my_path, '../data/processed/training/training_Y.npy'))
    X_test = np.load(os.path.join(my_path, '../data/processed/test/test_X.npy'))
    Y_test = np.load(os.path.join(my_path, '../data/processed/test/test_Y.npy'))


    nb_train_samples, img_rows, img_cols,img_channels = X_train.shape
    input_shape = (img_rows, img_cols, img_channels)
    nb_test_samples = X_test.shape[0]

    print('Input shape:', input_shape)
    print(nb_train_samples, 'train samples')
    print(nb_test_samples, 'test samples')

    input_shape = (img_rows, img_cols, img_channels)

    model = Sequential()
    model.add(Conv2D(64, kernel_size, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()    

    model.compile(loss=keras.losses.categorical_crossentropy,
          optimizer='adam',
          metrics=[angle_error])

    Y_train = keras.utils.to_categorical(Y_train, nb_classes)
    Y_test = keras.utils.to_categorical(Y_test, nb_classes)    

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, Y_test))

    # serialize model to JSON
    model_json = model.to_json()
    with open('models/cifar10.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5        
    model.save_weights('models/cifar10.h5')    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', type=str, default='mel', help='Choose from mel or mfcc model to classify.')
    # parser.add_argument('--file_path', type=str, default='data/cats_dogs/cat_1.wav', help='File you want to analyse.')
    # FLAGS, unparsed = parser.parse_known_args()
    # main(FLAGS)
    train_model()