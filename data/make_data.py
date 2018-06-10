from __future__ import print_function

import numpy as np
import argparse, sys, os, warnings

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import rotate_image, crop_around_center, binarize_images

# https://github.com/keras-team/keras/blob/a379b4207ab98c7e6b11ceb0e012a348d7b951d5/keras/datasets/cifar10.py
from keras.datasets import cifar10, mnist

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS=None

def process_image(image, angle):
    image_h, image_w = image.shape[0], image.shape[1]
    image_rotated = rotate_image(image, angle)
    image_rotated = crop_around_center(image_rotated, image_h, image_w)
    processed_image = np.vstack((image,image_rotated))

    return processed_image

def make_data(_):

    np.random.seed(FLAGS.seed)

    if FLAGS.dataset == 'cifar10':
        (x_train, _), (x_test, _) = cifar10.load_data()
    elif FLAGS.dataset == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()

    temp_x, temp_y = [], []

    print(x_train.shape)
    # Training set processing and saving
    for idx in range(x_train.shape[0]):
        angle = np.random.randint(360)
        if FLAGS.dataset == 'cifar10':
            processed_image = process_image(x_train[idx,:,:,:], angle)
        elif FLAGS.dataset == 'mnist':
            processed_image = process_image(x_train[idx,:,:], angle)
        temp_x.append(processed_image)
        temp_y.append(angle)

    temp_x = np.stack(temp_x, axis=0)
    temp_x = temp_x.astype(float)
    temp_x = binarize_images(temp_x)
    temp_y = np.stack(temp_y, axis=0)

    np.save('data/processed/training/training_X.npy', temp_x)
    np.save('data/processed/training/training_Y.npy', temp_y)

    # Test set processing and saving
    temp_x, temp_y = [], []
    for idx in range(x_test.shape[0]):
        angle = np.random.randint(360)
        if FLAGS.dataset == 'cifar10':
            processed_image = process_image(x_test[idx,:,:,:], angle)
        elif FLAGS.dataset == 'mnist':
            processed_image = process_image(x_test[idx,:,:], angle)
        temp_x.append(processed_image)
        temp_y.append(angle)

    temp_x = np.stack(temp_x, axis=0)
    temp_x = temp_x.astype(float)
    temp_x = binarize_images(temp_x)
    temp_y = np.stack(temp_y, axis=0)

    np.save('data/processed/test/test_X.npy', temp_x)
    np.save('data/processed/test/test_Y.npy', temp_y)

    print('files processed and data saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='Choose from cifar10 or MNIST dataset to create training and test datasets')
    parser.add_argument('--seed', type=int, default=0, help='pick seed')
    FLAGS, unparsed = parser.parse_known_args()
    make_data(FLAGS)
