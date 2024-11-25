from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D,
                                     BatchNormalization, Dense, Activation,
                                     Flatten, Dropout)
from tensorflow.keras.optimizers import SGD, Adam
# img shape is (463, 555, 3)
img_resize = 128
num_channels = 3
img_shape = (img_resize, img_resize, num_channels)

# Load the VGG19 model
base_model = VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=img_shape,
)

# Freezing the layers
for layer in base_model.layers:
    layer.trainable = False
# Sets the trainable attribute of each layer in the pre-trained model to False.
# This prevents the weights of these layers from being updated during backpropagation.
# This  retains the pre-trained features of VGG19 and avoid overwriting them with random gradients, especially if your dataset is small

# Define and compile the model
def VGG19():
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 Regularization
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # L2 Regularization
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model