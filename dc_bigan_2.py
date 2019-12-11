# Second version of BiGAN.

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import cv2

class BIGAN():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1#2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 5000

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=self.img_shape)
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        input_img = Input(shape=self.img_shape)
        
        img = Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same")(input_img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(32, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)        
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)

        img = Flatten()(img)
        
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        #img = Dropout(.3))
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        h = LeakyReLU(alpha=0.2)(img)

        z = Dense(self.latent_dim)(h)
        return Model(input_img, z)

    def build_generator(self):
        
        noise = Input(shape=(self.latent_dim,))
        
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128 * 20 * 20))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((20, 20, 128)))
        model.add(Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (3, 3), strides=2, padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("tanh"))

        #model.summary()
        
        img = model(noise)

        return Model(noise, img)
        
    def build_discriminator(self):
        
        z = Input(shape=(self.latent_dim, ))
        input_img = Input(shape=self.img_shape)
                
        img = Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same")(input_img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(32, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)        
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)

        img = Flatten()(img)
        
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        #img = Dropout(.3))
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        
        d_in = concatenate([z, img])
        d_in = Dense(1024)(d_in)
        d_in = LeakyReLU(alpha=0.2)(d_in)
        validity = Dense(1, activation='sigmoid')(d_in)
        
        return Model([z, input_img], validity)

    def generate_keras_input(self, mode = 'train'):
        if mode == 'train':
            path = 'Data/singlecoil_train'
        elif mode == 'val':
            path = 'Data/singlecoil_val'
        elif mode == 'test':
            path = 'Data/singlecoil_test_v2'
        else:
            print('Invalid mode')
            return

        files = sorted(os.listdir(path))

        for file in files:
            filename = os.path.join(path, file)
            hf = h5py.File(filename)

            mri = hf['reconstruction_rss'][()]
            kspace = np.fft.fft2(mri)
            kspace = np.fft.fftshift(kspace, (1,2))

            mri = mri[10:26, :, :]
            x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
            # add slight noise to image
            x = x + np.random.normal(0, .001, x.shape)
            x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
            
            batch_size = x.shape[0]
            
            x_down = np.zeros((batch_size, 160, 160))
                
            for i in range(batch_size):
                x_down[i, :, :] = cv2.resize(x[i, :, :], (160, 160))

            x = x_down

            x = x[:, :, :, np.newaxis]
            
            z = np.random.normal(size=(batch_size, self.latent_dim))

            yield (x, z)
        
        
if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=40000, batch_size=32, sample_interval=400)
