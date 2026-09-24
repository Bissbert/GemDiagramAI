import constants
import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, Conv2D, ZeroPadding2D, Concatenate
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

def build_generator(z_dim, meta_dim=0, image_size=constants.IMAGE_SIZE):

    # With meta_dim > 0 the generator takes [noise, metadata] and conditions
    # on the metadata by concatenating it to the noise vector.
    z = Input(shape=(z_dim,), name="z")
    inputs = [z]
    x = z
    if meta_dim:
        meta = Input(shape=(meta_dim,), name="metadata")
        inputs.append(meta)
        x = Concatenate()([z, meta])

    # Fully connected layer
    x = Dense(image_size//4 * image_size//4 * 128, activation="relu")(x)
    x = Reshape((image_size//4, image_size//4, 128))(x)

    # Upsample to image_size/2
    x = UpSampling2D()(x)
    x = Conv2D(128, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)

    # Upsample to image_size
    x = UpSampling2D()(x)
    x = Conv2D(64, kernel_size=3, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)

    x = Conv2D(channels, kernel_size=3, padding="same")(x)
    img = Activation("tanh")(x)

    return Model(inputs, img, name="generator")

def build_combined(generator, discriminator):

    # The combined model (stacked generator and discriminator) takes
    # noise (and metadata) as input => generates images => determines validity

    # For the combined model we will only train the generator
    discriminator.trainable = False

    inputs = [Input(shape=t.shape[1:]) for t in generator.inputs]
    img = generator(inputs if len(inputs) > 1 else inputs[0])
    if len(discriminator.inputs) > 1:
        # The discriminator judges the image against the same metadata
        validity = discriminator([img, inputs[1]])
    else:
        validity = discriminator(img)
    model = Model(inputs, validity, name="combined")

    # Add optimiser to discriminator and compile
    optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

def build_discriminator(img_shape, meta_dim=0):

    img = Input(shape=img_shape, name="image")
    inputs = [img]

    x = Conv2D(32, kernel_size=3, strides=2, padding="same")(img)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Flatten()(x)
    if meta_dim:
        # Judge the image together with the metadata it should match
        meta = Input(shape=(meta_dim,), name="metadata")
        inputs.append(meta)
        x = Concatenate()([x, meta])
    validity = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, validity, name="discriminator")

    # Add optimiser to discriminator and compile
    optimizer = tf.keras.optimizers.legacy.Adam(0.0002, 0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train(generator, discriminator, combined, imgs, metadata, epochs, batch_size, save_interval):

    # Labels for real and fake images
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    logging.info(f"Running training for {epochs} iterations")
    
    # Noise size and whether the models are conditioned on metadata
    z_dim = generator.inputs[0].shape[-1]
    conditioned = len(generator.inputs) > 1
    if conditioned:
        metadata = np.asarray(metadata, dtype=np.float32)
        if metadata.shape != (imgs.shape[0], generator.inputs[1].shape[-1]):
            raise ValueError(f"metadata shape {metadata.shape} does not match "
                             f"{imgs.shape[0]} images x {generator.inputs[1].shape[-1]} fields")

    # Setting up logging for the TensorBoard. histogram_freq=0: a weight
    # histogram of the generator's first Dense kernel (over 200 million
    # values) needs a [values, 30] bucket tensor and runs out of memory.
    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images and their corresponding metadata
        idx = np.random.randint(0, imgs.shape[0], batch_size)
        imgs_real = imgs[idx]
        meta_real = metadata[idx] if conditioned else None

        # Generate a batch of new images for the same metadata
        z = np.random.normal(0, 1, (batch_size, z_dim))
        imgs_fake = generator.predict([z, meta_real] if conditioned else z, verbose=0)

        # Train the discriminator
        d_in_real = [imgs_real, meta_real] if conditioned else imgs_real
        d_in_fake = [imgs_fake, meta_real] if conditioned else imgs_fake
        d_loss_real = discriminator.fit(d_in_real, real, epochs=1, verbose=0, callbacks=[tensorboard_callback])
        d_loss_fake = discriminator.fit(d_in_fake, fake, epochs=1, verbose=0, callbacks=[tensorboard_callback])

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
        z = np.random.normal(0, 1, (batch_size, z_dim))
        g_in = [z, metadata[np.random.randint(0, imgs.shape[0], batch_size)]] if conditioned else z
        g_loss = combined.fit(g_in, real, epochs=1, verbose=0, callbacks=[tensorboard_callback])

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

