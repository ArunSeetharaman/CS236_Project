# Evaluation for Poster.

import h5py
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective
from tqdm import tqdm

from keras import backend as K

from dc_bigan_2 import BIGAN

def undersample(imgs, factor):
    kspace2 = np.fft.fft2(imgs)
    kspace2 = np.fft.fftshift(kspace2, (1,2))
    
    mask = np.zeros(imgs.shape)
    mask[:,:,::factor,:] = 1
    kspace2 = mask * kspace2
    
    kspace2 = np.fft.ifftshift(kspace2, (1,2))
    under = np.fft.ifft2(kspace2)
    under = np.absolute(under)
    under = (under-np.min(under))/(np.max(under)-np.min(under))
    
    return under

if __name__ == '__main__':
    under = True
    factor = 4
    
    encoder = load_model('bigan_2/Models/encoder_24.h5')
    generator = load_model('bigan_2/Models/decoder_24.h5')

    bigan = BIGAN()

    train_generator = bigan.generate_keras_input('train')

    mse = []
    ssim = []

    for i in tqdm(range(973)):
        imgs, _ = next(train_generator)
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)
            
        z = encoder.predict(imgs)

        imgs_rec = generator.predict(z)

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))

    train_generator = bigan.generate_keras_input('val')

    mse = []
    ssim = []

    for i in tqdm(range(100)):
        imgs, _ = next(train_generator)
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)
            
        z = encoder.predict(imgs)

        imgs_rec = generator.predict(z)

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))


    from dc_aae_2 import AdversarialAutoencoder

    encoder = load_model('aae_huber_ssim_2/Models/encoder_aae_ssim_24.h5')
    generator = load_model('aae_huber_ssim_2/Models/decoder_aae_ssim_24.h5')

    aae = AdversarialAutoencoder()

    train_generator = aae.generate_keras_input('train')

    mse = []
    ssim = []

    for i in tqdm(range(973)):
        imgs, _ = next(train_generator)
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)

        z = encoder.predict(imgs)

        imgs_rec = generator.predict(z)

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))

    train_generator = aae.generate_keras_input('val')

    mse = []
    ssim = []

    for i in tqdm(range(100)):
        imgs, _ = next(train_generator)
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)
            
        z = encoder.predict(imgs)

        imgs_rec = generator.predict(z)

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))
    
    from dc_vae_2 import VariationalAutoencoder

    encoder = load_model('vae_huber_ssim_2/Models/encoder_vae_24.h5')
    generator = load_model('vae_huber_ssim_2/Models/decoder_vae_24.h5')

    vae = VariationalAutoencoder()

    train_generator = vae.generate_keras_input('train')

    mse = []
    ssim = []

    for i in tqdm(range(973)):
        imgs, _ = next(train_generator)[0]
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)
        
        params = encoder.predict(imgs)

        mu = params[:, :vae.latent_dim]
        log_var = params[:, vae.latent_dim:]

        z = (np.random.normal() * np.exp(.5 * log_var) + mu)

        imgs_rec = generator.predict(z)

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))

    train_generator = vae.generate_keras_input('val')

    mse = []
    ssim = []

    for i in tqdm(range(100)):
        imgs, _ = next(train_generator)[0]
        imgs_true = imgs
        if under:
            imgs = undersample(imgs, factor)
            
        params = encoder.predict(imgs)

        mu = params[:, :vae.latent_dim]
        log_var = params[:, vae.latent_dim:]

        z = (np.random.normal() * np.exp(.5 * log_var) + mu)

        imgs_rec = generator.predict(z)   

        mse.append(np.mean((imgs_true - imgs_rec)**2))

        imgs_true = K.constant(imgs_true)
        imgs_rec = K.constant(imgs_rec)

        ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

    print(np.mean(np.asarray(mse)))
    print(np.mean(np.asarray(ssim)))
