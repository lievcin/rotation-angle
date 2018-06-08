from __future__ import print_function

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import rotate_image, crop_around_center

# https://github.com/keras-team/keras/blob/a379b4207ab98c7e6b11ceb0e012a348d7b951d5/keras/datasets/cifar10.py
from keras.datasets import cifar10

def process_image(image, angle):
    image_h, image_w = image.shape[0], image.shape[1]
    image_rotated = rotate_image(image, angle)
    image_rotated = crop_around_center(image_rotated, image_h, image_w)
    processed_image = np.vstack((image,image_rotated))

    return processed_image

def make_data(seed=0):

    np.random.seed(seed)
    (x_train, _), (x_test, _) = cifar10.load_data()
    temp_x, temp_y = [], []

    # Training set processing and saving
    for idx in range(x_train.shape[0]):
        angle = np.random.randint(360)
        processed_image = process_image(x_train[idx,:,:,:], angle)
        temp_x.append(processed_image)
        temp_y.append(angle)

    temp_x = np.stack(temp_x, axis=0)
    temp_y = np.stack(temp_y, axis=0)

    np.save('data/processed/training/training_X.npy', temp_x)
    np.save('data/processed/training/training_Y.npy', temp_y)

    # Test set processing and saving
    temp_x, temp_y = [], []
    for idx in range(x_test.shape[0]):
        angle = np.random.randint(360)
        processed_image = process_image(x_test[idx,:,:,:], angle)
        temp_x.append(processed_image)
        temp_y.append(angle)

    temp_x = np.stack(temp_x, axis=0)
    temp_y = np.stack(temp_y, axis=0)

    np.save('data/processed/test/test_X.npy', temp_x)
    np.save('data/processed/test/test_Y.npy', temp_y)

    print('files processed and data saved')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_type', type=str, default='mel', help='Choose from mel or mfcc model to classify.')
    # parser.add_argument('--file_path', type=str, default='data/cats_dogs/cat_1.wav', help='File you want to analyse.')
    # FLAGS, unparsed = parser.parse_known_args()
    # main(FLAGS)
    make_data()
