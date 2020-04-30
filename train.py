"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
import os
import numpy as np
from datetime import datetime
import glob
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt

import tensorflow as tf

from parameters import *


################################################################################
# get data generator
def get_data_gen():
    # augment dataset using tf.keras.preprocessing.image.ImageDataGenerator
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # degrees
        horizontal_flip=True,
        rescale=1. / 255,
    )

    data_gen = image_generator.flow_from_directory(
        directory=os.path.join(os.getcwd(), "data"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        class_mode=None
    )

    return data_gen


################################################################################
# Main
if __name__ == "__main__":
    # print TF version
    print(f'TF version: {tf.__version__}')

    # create output directory for results
    output_dir = "results\\" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    train_data_gen = get_data_gen()
