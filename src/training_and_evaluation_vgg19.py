###################################################################
#   Project: Breast Cancer Classification
###################################################################
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# import functions from utils.py
from utils import generate_data_paths_with_label
from utils import fit_evaluate
from utils import visualize_model_performance

from model_definition_vgg19 import VGG19



def main():

    #Log file
    logging.basicConfig(filename='training_vgg19.log', level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # To check tensorflow GPU availability
    logging.info("Tensorflow version: {}".format(tf.__version__))
    logging.info("GPU available: {}".format(tf.config.list_physical_devices('GPU')))

    # Generate data paths with labels
    data_directory = '/path/to/data_dir'
    df = generate_data_paths_with_label(data_directory)
    logging.info('******************************************************************************')
    logging.info('Printing dataframe that contains image file paths and the corresponding labels')
    logging.info(df.to_string())
    logging.info('******************************************************************************')
    print('Columns in df: {}'.format(df.columns))

    logging.info('The total number of images: {}'.format(df.shape[0]))
    # Find counts for each label
    logging.info('Image counts per class')
    label_counts = df['labels'].value_counts()
    logging.info(label_counts)

    '''
    #############################################
    #               VISUALIZATION
    #############################################
    # To load a single image
    img = load_img(df['filepaths'].iloc[0])
    img_array = img_to_array(img)
    img_normalized = img_array/255
    print(img_array.shape)  # image size : (463, 555, 3)
    print(img_array[:, :, 0])
    print(img_array[:, :, 1])
    print(img_array[:, :, 2])
    plt.imshow(img_normalized)
    plt.show()
    ############################################
    '''
    ############################################
    #   Data preparation
    ############################################
    # img shape is (463, 555, 3)
    img_resize = 128
    num_channels = 3
    img_shape = (img_resize, img_resize, num_channels)
    # Define the images and labels list
    images = []
    labels = []
    # Get the class names
    class_names = df['labels'].unique()
    print(f"class_names: {class_names}")
    for index, row in df.iterrows():
        print(f"index: {index}")
        print(f"file path: {row['filepaths']}")
        print(f"label: {row['labels']}")
        img = load_img(row['filepaths'], target_size=(img_resize, img_resize))
        # Load image array and Normalize
        img_array = img_to_array(img) / 255.0
        print(f"Shape of image array: {img_array.shape}")
        images.append(img_array)
        labels.append(row['labels'])


    # Convert to numpy array
    x = np.array(images)
    print(f"Shape of x: {x.shape}")
    y = np.array(labels)
    print(f"Shape of y: {y.shape}")
    print(f"y : {y}")

    # One-hot encode labels
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=42)
    print(f"Training set shape: {x_train.shape}, Testing set shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}, Testing labels shape: {y_test.shape}")

    # To find the number of instances per class in x_train and x_test when y contains one-hot encoded labels
    # First decode the one-hot labels back to their original class labels and then count
    from collections import Counter
    # Decode one-hot labels to class indices
    y_train_classes = np.argmax(y_train, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    # Ensure the class indices are in a suitable format
    y_train_classes = np.asarray(y_train_classes).flatten()  # Convert to a flat array
    y_test_classes = np.asarray(y_test_classes).flatten()
    # Count instances per class
    train_counts = Counter(y_train_classes)
    test_counts = Counter(y_test_classes)
    print("Instances per class in x_train:", dict(train_counts))
    print("Instances per class in x_test:", dict(test_counts))

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],  # Adjust brightness
        fill_mode='nearest'
    )
    # Fit the generator to the training data
    datagen.fit(x_train)

    # Instantiate model
    model = VGG19()
    model.summary()

    # Train the model
    fit_evaluate(encoder, model, x_train, y_train, x_test, y_test, bs=8, Epochs=150, patience=4)

    # Visualize predictions on a subset of test images
    visualize_model_performance(model, x_test, y_test)



if __name__ == '__main__':
    main()