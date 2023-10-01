import constants
import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import TensorBoard


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
np.random.seed(1000)

# Image dimensions

img_cols = constants.IMAGE_SIZE
channels = 3
img_shape = (constants.IMAGE_SIZE, img_cols, channels)

# Size of the noise vector, used as input to the Generator
z_dim = 100

def build_generator(z_dim):

    model = Sequential()

    # Fully connected layer
    model.add(Dense(constants.IMAGE_SIZE//4 * constants.IMAGE_SIZE//4 * 128, activation="relu", input_dim=z_dim))
    model.add(Reshape((constants.IMAGE_SIZE//4, constants.IMAGE_SIZE//4, 128)))

    # Upsample to 256x256
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # Upsample to 512x512
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    return model

def build_combined(generator, discriminator):

    # The combined model (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity

    # For the combined model we will only train the generator
    discriminator.trainable = False

    model = Sequential([generator, discriminator])
    
    # Add optimiser to discriminator and compile
    optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

def build_discriminator(img_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    # Add optimiser to discriminator and compile
    optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train(generator, discriminator, combined, imgs, metadata, epochs, batch_size, save_interval):

    # Labels for real and fake images
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    logging.info(f"Running training for {epochs} iterations")
    
    # Setting up logging for the TensorBoard
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images and their corresponding metadata
        idx = np.random.randint(0, imgs.shape[0], batch_size)
        imgs_real = imgs[idx]
        
        # Generate a batch of new images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        imgs_fake = generator.predict(z)

        # Train the discriminator
        d_loss_real = discriminator.fit(imgs_real, real, epochs=1, verbose=0, callbacks=[tensorboard_callback])
        d_loss_fake = discriminator.fit(imgs_fake, fake, epochs=1, verbose=0, callbacks=[tensorboard_callback])

        d_loss_real_val = d_loss_real.history['loss'][0]
        d_loss_fake_val = d_loss_fake.history['loss'][0]
        d_loss_avg = 0.5 * (d_loss_real_val + d_loss_fake_val)
        
        d_acc_real = d_loss_real.history['accuracy'][0] if 'accuracy' in d_loss_real.history else None
        d_acc_fake = d_loss_fake.history['accuracy'][0] if 'accuracy' in d_loss_fake.history else None
        d_acc_avg = 0.5 * (d_acc_real + d_acc_fake) if d_acc_real is not None and d_acc_fake is not None else None

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator using the combined model
        z = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.fit(z, real, epochs=1, verbose=0, callbacks=[tensorboard_callback])

        # Print the progress
        if d_acc_avg is not None:
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss_avg, 100*d_acc_avg, g_loss.history['loss'][0]))
        else:
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss_avg, g_loss.history['loss'][0]))

        # If at save interval => save generated image samples and print the progress
        if epoch % save_interval == 0:
            model_save_path = f"generator_model_epoch_{epoch}.h5"
            generator.save(model_save_path)
            logging.info(f"Saved generator model at epoch {epoch} to {model_save_path}")
            logging.info(f"{epoch} [D loss: {d_loss_avg} | D accuracy: {100 * d_acc_avg if d_acc_avg is not None else 'N/A'}] [G loss: {g_loss.history['loss'][0]}]")
            

    return generator

