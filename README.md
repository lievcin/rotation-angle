# find-rotation-angle
==============================

This is a small project to find the rotation angle of a given image using neural network

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modelling.
    │   └── make_data.py   <- Script to generate the training and test data.
    │                         can be executed as python data/make_data.py and accepts parameters dataset=mnist or cifar10
    │                         and seed which should be an integer. By default makes data with cifar10 and seed=0
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │   ├── train_model.py <- Script to train a model in 50 epochs.
    │   │                      can be executed as python models/train_model.py accepts parameters dataset=mnist or cifar10
    │   │                      which depends on the contents in the data/processed folder, as it will not train correctly or
    │   │                      at all if the datasets don't match.
    │   │── cifar10.h5     <- pre-trained model for 50 epochs of cifar10 data
    │   │── cifar10.json   <- pre-trained model for 50 epochs of cifar10 data
    │   │── mnist.h5       <- pre-trained model for 100 epochs of mnist data
    │   └── mnist.json     <- pre-trained model for 100 epochs of mnist data
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering).
    │                         In these notebooks there's some data exploration regarding downloading the files,
    │                         as well as rotating images and generating sample images for test through console
    │
    ├── test_images        <- Two pairs of images that can be used to test the classification models.
    │                         In these notebooks there's some data exploration regarding downloading the files,
    │                         as well as rotating images and generating sample images for test through console    
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── utils.py           <- utility functions file used across the project
    │
    └── test_models.py     <- Script to test the models on test images or provided ones. It can take three arguments
                              original_image_path, rotated_image_path and model_name (by default cifar10). The script will
                              check the sizes of both images and then classify using the supplied model.
