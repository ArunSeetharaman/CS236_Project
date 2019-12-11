from __future__ import print_function, division

from keras.layers import Conv2D, Dense, AveragePooling2D, Activation, Cropping2D, Dropout, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Flatten, Input, add, Lambda, concatenate, LeakyReLU, multiply
from keras.layers import GlobalAveragePooling2D, average
from keras.models import model_from_json, Model
from keras.initializers import VarianceScaling
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
        out = AveragePooling2D()(out)

    return out

class BIGAN():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1#2
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        optimizer = Adam(lr = 0.0001, decay = 0.00001)

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
        under = Input(shape=self.img_shape)
        
        z_ = self.encoder(under)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, under, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        inp = Input(shape = self.img_shape)

        x = d_block(inp, 1 * cha)   #64
        x = d_block(x, 2 * cha)   #32
        x = d_block(x, 3 * cha)   #16
        x = d_block(x, 4 * cha)  #8
        x = d_block(x, 8 * cha)  #4
        x = d_block(x, 16 * cha, p = False)  #4

        x = Flatten()(x)

        x = Dense(16 * cha, kernel_initializer = 'he_normal')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(self.latent_dim, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)

        return Model(inputs = inp, outputs = x)

    def build_generator(self):
        
        #Inputs
        inp = Input(shape = (self.latent_dim,))

        #Latent

        #Actual Model
        x = Dense(5*5*16*cha, kernel_initializer = 'he_normal')(inp)
        x = Reshape([5, 5, 16*cha])(x)

        x = g_block(x, 16 * cha, u = False)  #5
        x = g_block(x, 8 * cha)  #10
        x = g_block(x, 4 * cha)  #20
        x = g_block(x, 3 * cha)   #40
        x = g_block(x, 2 * cha)   #80
        x = g_block(x, 1 * cha)   #160

        x = Conv2D(filters = 1, kernel_size = 1, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(x)

        return Model(inputs = inp, outputs = x)

        
    def build_discriminator(self):
        
        inp = Input(shape = self.img_shape)
        inpl = Input(shape = (self.latent_dim,))

        #Latent input
        l = Dense(512, kernel_initializer = 'he_normal')(inpl)
        l = LeakyReLU(0.2)(l)
        l = Dense(512, kernel_initializer = 'he_normal')(l)
        l = LeakyReLU(0.2)(l)
        l = Dense(512, kernel_initializer = 'he_normal')(l)
        l = LeakyReLU(0.2)(l)

        x = d_block(inp, 1 * cha)   #64
        x = d_block(x, 2 * cha)   #32
        x = d_block(x, 3 * cha)   #16
        x = d_block(x, 4 * cha)  #8
        x = d_block(x, 8 * cha)  #4
        x = d_block(x, 16 * cha, p = False)  #4

        x = Flatten()(x)

        x = concatenate([x, l])

        x = Dense(16 * cha, kernel_initializer = 'he_normal')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(1, activation = 'sigmoid', kernel_initializer = 'he_normal')(x)

        return Model(inputs = [inpl, inp], outputs = x)

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
        
        
if __name__ == '__main__':
    bigan = BIGAN()
    bigan.train(epochs=40000, batch_size=32, sample_interval=400)
