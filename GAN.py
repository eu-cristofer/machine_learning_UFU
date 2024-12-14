# -*- coding: utf-8 -*-
# Import necessary libraries
import tensorflow as tf
import glob
import imageio  # For creating GIFs
import matplotlib.pyplot as plt  # For plotting images
import numpy as np
import os  # For file handling
from PIL import Image  # For image manipulation
from tensorflow.keras import layers  # For building neural network layers
import time  # For tracking time
from IPython import display  # For clearing and updating outputs in Jupyter Notebooks

# Check and print the TensorFlow version being used
print(f"TensorFlow version: {tf.__version__}")

# Load the MNIST dataset (handwritten digit images)
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

# Preprocess the images:
# - Reshape to include a channel dimension (for grayscale images)
# - Normalize pixel values to the range [-1, 1]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize

# Define constants for batching and shuffling the dataset
BUFFER_SIZE = 60000  # Size of the dataset
BATCH_SIZE = 256  # Number of images per training batch

# Create a TensorFlow dataset, shuffle, and batch it
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),  # Input noise vector
        layers.BatchNormalization(),  # Normalize activations
        layers.LeakyReLU(),  # Activation function
        layers.Reshape((7, 7, 256)),  # Reshape to a feature map
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),  # Upsample
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),  # Upsample
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')  # Output layer
    ])
    return model

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),  # Downsample
        layers.LeakyReLU(),
        layers.Dropout(0.3),  # Prevent overfitting
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),  # Further downsampling
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),  # Flatten for dense layers
        layers.Dense(1)  # Output a single value (real/fake probability)
    ])
    return model

# Define loss functions for the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Loss for the discriminator
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)  # Compare real images with label 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # Compare fake images with label 0
    return real_loss + fake_loss

# Loss for the generator
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)  # Compare fake images with label 1

# Define optimizers for both models
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Instantiate the generator and discriminator
generator = make_generator_model()
discriminator = make_discriminator_model()

# Set up checkpointing to save and restore model states during training
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Training parameters
EPOCHS = 50
noise_dim = 100  # Dimension of the random noise vector
num_examples_to_generate = 16  # Number of samples to generate for visualization
seed = tf.random.normal([num_examples_to_generate, noise_dim])  # Fixed seed for consistency in generated samples

# Function to perform a single training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])  # Generate random noise
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)  # Generate fake images
        real_output = discriminator(images, training=True)  # Discriminator's response to real images
        fake_output = discriminator(generated_images, training=True)  # Discriminator's response to fake images
        gen_loss = generator_loss(fake_output)  # Calculate generator loss
        disc_loss = discriminator_loss(real_output, fake_output)  # Calculate discriminator loss
    # Compute gradients and apply them
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)  # Generate images from fixed seed
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')  # Denormalize and plot
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')  # Save the plot as an image
    plt.close()

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()  # Track epoch start time
        for image_batch in dataset:
            train_step(image_batch)  # Train on each batch
        display.clear_output(wait=True)  # Clear previous output
        generate_and_save_images(generator, epoch + 1, seed)  # Save generated images
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)  # Save checkpoint every 15 epochs
        print(f'Time for epoch {epoch + 1} is {time.time()-start:.2f} sec')  # Print epoch duration
    display.clear_output(wait=True)  # Clear output after training
    generate_and_save_images(generator, epochs, seed)  # Save final images

# Train the GAN
train(train_dataset, EPOCHS)

# Restore the latest checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Create an animated GIF of the generated images
anim_file = 'dcgan.gif'
with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = sorted(glob.glob('image_at_epoch_*.png'))  # Load generated image filenames
    for filename in filenames:
        image = imageio.imread(filename)  # Read each image
        writer.append_data(image)  # Append to the GIF
    writer.append_data(imageio.imread(filenames[-1]))  # Add the last frame again for consistency

print(f"GIF saved as {anim_file}.")
