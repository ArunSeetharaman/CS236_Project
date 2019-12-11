# Second version of VAE.

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
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

class VariationalAutoencoder():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 5000

        optimizer = Adam()#(0.0002, 0.5)

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        prior_params = Input(shape=(2*self.latent_dim,))
        
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        params = self.encoder(img)
        
        mu = params[:, 0:self.latent_dim]
        log_var = params[:, self.latent_dim:]
        
        latent_repr = Lambda(lambda p: p[:, 0:self.latent_dim] + K.random_normal(K.shape(p[:, 0:self.latent_dim])) * K.exp(.5 *p[:, self.latent_dim:]), output_shape=(self.latent_dim,))(params)
                
        reconstructed_img = self.decoder(latent_repr)
        
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.variational_autoencoder = Model([img, prior_params], [reconstructed_img, params])
        
        self.variational_autoencoder.compile(loss=[reconstruction, dkl_normal],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        input_img = Input(shape=self.img_shape)
        
        img = Conv2D(32, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same")(input_img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(32, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        #img = Dropout(.3))
        
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(64, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        #img = Dropout(.3)(img)
        
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)        
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(128, kernel_size=3, strides=1, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        #img = Dropout(.3)(img)
        '''
        img = Conv2D(128, kernel_size=3, strides=2, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(128, kernel_size=3, strides=2, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(img)
        #img = Dropout(.3)(img)
        
        img = Conv2D(128, kernel_size=3, strides=2, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        img = Conv2D(128, kernel_size=3, strides=2, padding="same")(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        #img = Dropout(.3)(img)
        '''
        img = Flatten()(img)
        
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        img = LeakyReLU(alpha=0.2)(img)
        #img = Dropout(.3))
        img = Dense(1024)(img)
        img = BatchNormalization(momentum=0.8)(img)
        h = LeakyReLU(alpha=0.2)(img)
                        
        #mu = Dense(self.latent_dim)(h)
        #var = Dense(self.latent_dim)(h)
        
        params = Dense(2 * self.latent_dim)(h)

        return Model(input_img, params) #latent_repr)

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
        #model.add(UpSampling2D())
        model.add(Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=False))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(.3))
        '''
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Conv2DTranspose(1, (3, 3), strides=2, padding='same', use_bias=False))
        #model.add(Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
        model.add(UpSampling2D())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(.3))
        '''
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(128, (3, 3), strides=2, padding='same', use_bias=False))
        #model.add(UpSampling2D())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(.3))
        
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2DTranspose(64, (3, 3), strides=2, padding='same', use_bias=False))
        #model.add(UpSampling2D())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(.3))
        
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(.3))
        
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("tanh"))

        #model.summary()
        
        img = model(noise)

        return Model(noise, img)

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
        while(1):
            for file in files:
                filename = os.path.join(path, file)
                hf = h5py.File(filename)

                mri = hf['reconstruction_rss'][()]
                kspace = np.fft.fft2(mri)
                kspace = np.fft.fftshift(kspace, (1,2))

                mri = mri[10:26, :, :]
                #x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
                # add slight noise to image
                #x = x + np.random.normal(0, .001, x.shape)
                x = 2*(mri-np.min(mri))/(np.max(mri)-np.min(mri))-1
                #x = (mri - np.mean(mri))/(np.std(mri))
                
                batch_size = x.shape[0]
                
                x_down = np.zeros((batch_size, 160, 160))
                
                for i in range(batch_size):
                    x_down[i, :, :] = cv2.resize(x[i, :, :], (160, 160))
                
                x = x_down
                
                x = x[:, :, :, np.newaxis]

                #idx = np.random.randint(0, x.shape[0], batch_size)
                #x= x[idx]

                mu = np.zeros((batch_size, self.latent_dim))
                var = np.ones((batch_size, self.latent_dim))
                params = np.concatenate((mu, var), -1)

                yield ([x, params], [x, params])

def dkl_normal(params_true, params_pred):
    mu_true = params_true[:, :5000]
    var_true = K.exp(params_true[:, 5000:])
    mu_pred = params_pred[:, :5000]
    var_pred = K.exp(params_pred[:, 5000:])
    
    #kl = .5 * (K.log(params_true[:, 1000:]) - K.log(params_pred[:, 1000:]) + params_pred[:, 1000:] / params_true[:, 1000:] + K.pow(params_pred[:, :1000] - params_true[:, :1000], 2) / params_pred[:, 1000:] - 1)
    kl = .5 * (K.log(var_true) - K.log(var_pred) + var_pred / var_true + K.pow(mu_pred - mu_true, 2) / var_pred - 1)
    #kl = .5 * (- K.log(var_pred) + var_pred + K.pow(mu_pred, 2) / var_pred - 1)

    kl = K.sum(kl, -1)
    kl = K.mean(kl, 0)
    
    return kl

def reconstruction(img, reconstructed_img):
    return .5 * huber_loss(img, reconstructed_img) + .5 * DSSIMObjective()(img, reconstructed_img)

# def log_normal():

if __name__ == '__main__':
    aae = variationalAutoencoder()
    aae.train(epochs=20000, batch_size=32, sample_interval=200)
