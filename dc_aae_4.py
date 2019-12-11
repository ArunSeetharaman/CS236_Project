from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Cropping2D, Dropout, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Flatten, Input, add, Lambda, concatenate, LeakyReLU, multiply
from keras.layers import GlobalAveragePooling2D, average
from keras.initializers import VarianceScaling
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

cha = 16

def g_block(inp, fil, u = True):

    if u:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
    else:
        out = Activation('linear')(inp)

    skip = Conv2D(fil, 1, padding = 'same', kernel_initializer = 'he_normal')(out)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.2)(out)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.2)(out)

    out = Conv2D(fil, 1, padding = 'same', kernel_initializer = 'he_normal')(out)

    out = add([out, skip])
    out = LeakyReLU(0.2)(out)

    return out

def d_block(inp, fil, p = True):

    skip = Conv2D(fil, 1, padding = 'same', kernel_initializer = 'he_normal')(inp)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.2)(out)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.2)(out)

    out = Conv2D(fil, 1, padding = 'same', kernel_initializer = 'he_normal')(out)

    out = add([out, skip])
    out = LeakyReLU(0.2)(out)

    if p:
        out = MaxPooling2D()(out)

    return out

class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128#64

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
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'], loss_weights = [.5, .5], optimizer=optimizer)
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
        
        #Inputs
        inp = Input(shape = [self.latent_dim])

        #Latent

        #Actual Model
        x = Dense(5*5*16*cha, kernel_initializer = 'he_normal')(inp)
        x = Reshape([5, 5, 16*cha])(x)

        x = g_block(x, 16 * cha, u = False)  #4
        x = g_block(x, 8 * cha)  #8
        x = g_block(x, 4 * cha)  #16
        x = g_block(x, 3 * cha)   #32
        x = g_block(x, 2 * cha)   #64
        x = g_block(x, 1 * cha)   #128

        x = Conv2D(filters = 1, kernel_size = 1, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(x)

        return Model(inputs = inp, outputs = x)

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
        factor = 4
        
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
            mri = mri[10:26, :, :]
            x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
            x = x[:, :, :, np.newaxis]
            
            batch_size = x.shape[0]
            
            x_down = np.zeros((batch_size, 160, 160))
                
            for i in range(batch_size):
                x_down[i, :, :] = cv2.resize(x[i, :, :], (160, 160))

            x = x_down[:,:,:,np.newaxis]
            
            kspace = np.fft.fft2(x)
            kspace = np.fft.fftshift(kspace, (1,2))
    
            mask = np.zeros(kspace.shape)
            mask[:,:,::factor] = 1
            kspace2 = mask * kspace

            kspace2 = np.fft.ifftshift(kspace2, (1,2))
            under = np.fft.ifft2(kspace2)
            under = np.absolute(under)
            under = 2*(under-np.min(under))/(np.max(under)-np.min(under))-1
            
            z = np.random.normal(size=(batch_size, self.latent_dim))

            yield (x, under, z)

def reconstruction(img, reconstructed_img):
    #return huber_loss(img, reconstructed_img)
    return DSSIMObjective()(img, reconstructed_img)
    #.5 * huber_loss(img, reconstructed_img) + .5 * DSSIMObjective()(img, reconstructed_img)

if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20000, batch_size=32, sample_interval=200)
