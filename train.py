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
from model import build_discriminator, build_generator


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


# plot images in a 1x5 grid
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# discriminaor loss function
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    total_loss = real_loss + fake_loss
    return total_loss


# generator loss function
def generator_loss(generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generated_loss = cross_entropy(
        tf.ones_like(generated_output),
        generated_output
    )

    return generated_loss


# generate and save images
def generate_and_save_images(model, epoch, z_input, save_dir):
    predictions = model(z_input, training=False)
    predictions = predictions[:16]  # generate 16 images

    # rescale from [-1, 1] to [0, 1]
    predictions = (predictions + 1) / 2

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis("off")

    fig_name = os.path.join(save_dir, f'Epoch {epoch:05d}')
    plt.savefig(fig_name)
    plt.close()


# training loop
def train(dataset, d, g, d_optimizer, g_optimizer, z_input, save_dir):
    # training metrics
    d_train_loss = tf.keras.metrics.Mean("d_train_loss", dtype=tf.float32)
    g_train_loss = tf.keras.metrics.Mean("g_train_loss", dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(save_dir)

    for e in range(NUM_EPOCHS):
        # get a batch
        num_batches = len(dataset)
        for i in range(num_batches):
            batch = dataset[i]

            # noise input for generator
            noise = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))

            # GradientTape
            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                # generator
                fake_batch = g(noise, training=True)

                # discriminator
                real_output = d(batch, training=True)
                fake_output = d(fake_batch, training=True)

                # loss functions
                g_loss = generator_loss(fake_output)
                d_loss = discriminator_loss(real_output, fake_output)

            # compute gradients recorded on "tape"
            g_gradients = g_tape.gradient(g_loss, g.trainable_variables)
            d_gradients = d_tape.gradient(d_loss, d.trainable_variables)

            # apply gradients to model variables to minimize loss function
            g_optimizer.apply_gradients(zip(g_gradients, g.trainable_variables))
            d_optimizer.apply_gradients(zip(d_gradients, d.trainable_variables))

            d_train_loss(d_loss)
            g_train_loss(g_loss)

        if e % 200 == 0 or e == NUM_EPOCHS - 1:
            # generate sample output
            generate_and_save_images(
                model=g,
                epoch=e+1,
                z_input=z_input,
                save_dir=save_dir
            )

        # write loss metrics to TensorBoard
        with train_summary_writer.as_default():
            tf.summary.scalar("d_loss", d_train_loss.result(), step=e)
            tf.summary.scalar("g_loss", g_train_loss.result(), step=e)

        # reset metrics every epoch
        d_train_loss.reset_states()
        g_train_loss.reset_states()


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

    """
    x = next(train_data_gen)
    print(len(x))
    plot_images(x[:5])
    """

    # ----- MODEL ----- #
    # discriminator
    discriminator = build_discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )
    discriminator.summary()

    # generator
    generator = build_generator()
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )
    generator.summary()

    # ----- TRAINING ----- #
    z_input_gen = tf.random.normal(shape=(BATCH_SIZE, NOISE_DIM))

    train(
        dataset=train_data_gen,
        d=discriminator,
        g=generator,
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        z_input=z_input_gen,
        save_dir=output_dir
    )

    # ----- SAVE ----- #
