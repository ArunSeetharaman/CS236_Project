# Second version of AAE.

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.losses import huber_loss
from keras_contrib.losses import DSSIMObjective

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import cv2

class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 5000 # used to be 100

        optimizer = Adam(0.0002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=[reconstruction, 'binary_crossentropy'], loss_weights = [.5, .5], optimizer=optimizer)
            #loss_weights=[0.99, 0.01],
            #optimizer=optimizer)


    def build_encoder(self):
        # Encoder

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
                        
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                lambda p: p[0])([mu, log_var])

        return Model(input_img, latent_repr)

    def build_decoder(self):
        
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
        
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Dense(1024))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

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
            '''
            kspace = hf['kspace'][()]
            if kspace.shape[-1] == 372:
                kspace = kspace[:, :, 2:-2]
            '''
            mri = hf['reconstruction_rss'][()]
            kspace = np.fft.fft2(mri)
            kspace = np.fft.fftshift(kspace, (1,2))
            '''
            x = np.zeros(kspace.shape + (2, ))
            
            x[:,:,:,0] = np.real(kspace)
            x[:,:,:,1] = np.imag(kspace)
            '''
            '''
            x = np.zeros((16, 320, 320, 2))
            x[:,:,:,0] = np.real(kspace[10:26, :, :])
            x[:,:,:,1] = np.imag(kspace[10:26, :, :])
            x = (x-np.mean(x, (0, 1, 2)))/(np.std(x, (0, 1, 2)))
            '''
            mri = mri[10:26, :, :]
            #x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
            # add slight noise to image
            #x = x + np.random.normal(0, .001, x.shape)
            x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
            x = x[:, :, :, np.newaxis]
            
            batch_size = x.shape[0]
            
            x_down = np.zeros((batch_size, 160, 160))
                
            for i in range(batch_size):
                x_down[i, :, :] = cv2.resize(x[i, :, :], (160, 160))

            x = x_down[:,:,:,np.newaxis]
            
            #idx = np.random.randint(0, x.shape[0], batch_size)
            #x= x[idx]
            
            z = np.random.normal(size=(batch_size, self.latent_dim))

            yield (x, z)

def reconstruction(img, reconstructed_img):
    return .5 * huber_loss(img, reconstructed_img) + .5 * DSSIMObjective()(img, reconstructed_img)

if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20000, batch_size=32, sample_interval=200)
