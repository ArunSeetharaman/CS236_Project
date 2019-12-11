import h5py
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective
from tqdm import tqdm

from keras import backend as K
'''
from dc_bigan import BIGAN

encoder = load_model('encoder_4.h5')
generator = load_model('generator_4.h5')

bigan = BIGAN()

train_generator = bigan.generate_keras_input('train')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    z = encoder.predict(imgs)
    
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))

#train_mse = np.concatenate(mse)
#train_ssim = np.concatenate(ssim)

print(mse)
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

train_generator = bigan.generate_keras_input('val')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    z = encoder.predict(imgs)
    
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))
    
#val_mse = np.concatenate(mse)
#val_ssim = np.concatenate(ssim)

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))
'''
'''
from dc_aae import AdversarialAutoencoder

encoder = load_model('encoder_aae.h5')
generator = load_model('decoder_aae.h5')

aae = AdversarialAutoencoder()

train_generator = aae.generate_keras_input('train')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    z = encoder.predict(imgs)
    
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

train_generator = aae.generate_keras_input('val')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    z = encoder.predict(imgs)
    
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))
'''
from dc_vae import VariationalAutoencoder

encoder = load_model('vae_huber_ssim/Models/encoder_vae_24.h5')
generator = load_model('vae_huber_ssim/Models/decoder_vae_24.h5')

vae = VariationalAutoencoder()

train_generator = vae.generate_keras_input('train')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    imgs = imgs[0]
    
    params = encoder.predict(imgs)
    
    mu = params[:, :vae.latent_dim]
    log_var = params[:, vae.latent_dim:]
    
    z = (np.random.normal() * np.exp(.5 * log_var) + mu)
    
    imgs_rec = generator.predict(z)
    
    imgs = (imgs-np.min(imgs))/(np.max(imgs)-np.min(imgs))
    imgs_rec = (imgs_rec-np.min(imgs_rec))/(np.max(imgs_rec)-np.min(imgs_rec))
    #imgs_rec = (imgs_rec-np.min(imgs))/(np.max(imgs)-np.min(imgs))

    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

train_generator = vae.generate_keras_input('val')

mse = []
ssim = []

for i in tqdm(range(20)):
    imgs, _ = next(train_generator)
    
    imgs = imgs[0]
    
    params = encoder.predict(imgs)
    
    mu = params[:, :vae.latent_dim]
    log_var = params[:, vae.latent_dim:]
    
    z = (np.random.normal() * np.exp(.5 * log_var) + mu)
    
    imgs_rec = generator.predict(z)
    
    imgs = (imgs-np.min(imgs))/(np.max(imgs)-np.min(imgs))
    imgs_rec = (imgs_rec-np.min(imgs_rec))/(np.max(imgs_rec)-np.min(imgs_rec))
    #imgs_rec = (imgs_rec-np.min(imgs))/(np.max(imgs)-np.min(imgs))    
    
    mse.append(np.mean((imgs - imgs_rec)**2))
    
    imgs = K.constant(imgs)
    imgs_rec = K.constant(imgs_rec)
    
    ssim.append(K.eval(DSSIMObjective()(imgs, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))