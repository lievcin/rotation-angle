import numpy as np
from utils import rotate_image, crop_around_center, binarize_images

import argparse, sys, os, warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.image as mpimg

import keras
from keras.models import model_from_json

FLAGS=None

def test_images(_):

    if os.path.exists(FLAGS.original_image_path) == False or os.path.exists(FLAGS.rotated_image_path) == False:
        print('Cannot find one or two of the images needed')
        sys.exit()
    else:
        original_img = mpimg.imread(FLAGS.original_image_path)
        rotated_img = mpimg.imread(FLAGS.rotated_image_path)

    if original_img.shape == rotated_img.shape:

        processed_image = np.vstack((original_img,rotated_img))

        input_h = processed_image.shape[0] #Height
        input_w = processed_image.shape[1] #Width
        input_d = processed_image.shape[2] #Depth

        processed_image = processed_image.reshape(1, input_h, input_w, input_d)

    else:
        print('Image sized don''t match')
        sys.exit()

    # load json and create model
    json_file = open('models/' + FLAGS.model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights('models/' + FLAGS.model_name + '.h5')

    scores = loaded_model.predict(processed_image)
    angle_difference = np.argmax(scores)

    print('Angle difference between the pictures: ' + str(angle_difference) + ' degrees.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_image_path', type=str, default='test_images/truck_original.jpg')
    parser.add_argument('--rotated_image_path', type=str, default='test_images/truck_rotated.jpg')
    parser.add_argument('--model_name', type=str, default='cifar10')
    FLAGS, unparsed = parser.parse_known_args()
    test_images(FLAGS)
