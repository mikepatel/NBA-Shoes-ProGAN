"""
Michael Patel
April 2020

Project description:
    Build a GAN to create basketball shoe designs

File description:
    For model and training parameters
"""
################################################################################
NUM_EPOCHS = 2000
BATCH_SIZE = 20  # 35

NOISE_DIM = 100

LEARNING_RATE = 0.0002
BETA_1 = 0.9  # 0.5

LEAKY_ALPHA = 0.3  # default is 0.3
DROPOUT_RATE = 0.3

IMAGE_WIDTH = 4
IMAGE_HEIGHT = 4
NUM_CHANNELS = 3
