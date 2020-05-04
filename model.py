"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model definitions
"""
################################################################################
# Imports
import os
import tensorflow as tf

from parameters import *


################################################################################
# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()

    # Layer: Conv: 8x8 (alpha path)

    # Layer: Downsample (1-alpha path)

    # Layer: Fade In

    # Layer: Conv: 4x4
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(5, 5),
        strides=1,
        input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
        padding="same"
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer: Dropout
    model.add(tf.keras.layers.Dropout(rate=DROPOUT_RATE))

    # Layer: Flatten
    model.add(tf.keras.layers.Flatten())

    # Layer: Output
    model.add(tf.keras.layers.Dense(
        units=1
    ))

    return model


################################################################################
# Generator
def build_generator():
    model = tf.keras.Sequential()

    # Layer: Fully connected
    model.add(tf.keras.layers.Dense(
        units=4*4*64,
        input_shape=(NOISE_DIM, ),
        use_bias=False
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=LEAKY_ALPHA))

    # Layer: Reshape
    model.add(tf.keras.layers.Reshape(
        target_shape=(4, 4, 64)
    ))

    # Conv higher resolution

    # Layer: Output: 4x4x3
    model.add(tf.keras.layers.Conv2DTranspose(
        filters=3,  # RGB
        kernel_size=(5, 5),
        strides=1,
        padding="same",
        activation=tf.keras.activations.tanh
    ))

    # Layer: alpha

    return model
