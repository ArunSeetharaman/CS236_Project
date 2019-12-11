import h5py
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective

from keras import backend as K

from dc_aae_3 import AdversarialAutoencoder

encoder = load_model('aae_mse_6/Models/encoder_aae_ssim_19.h5')
generator = load_model('aae_mse_6/Models/decoder_aae_ssim_19.h5')

aae = AdversarialAutoencoder()

train_generator = aae.generate_keras_input('train')#val')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(train_generator)
    z = encoder.predict(under)
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

val_generator = aae.generate_keras_input('val')#val')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(val_generator)
    z = encoder.predict(under)
    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))


from dc_vae_3 import VariationalAutoencoder

encoder = load_model('vae_mse_3/Models/encoder_vae_19.h5')
generator = load_model('vae_mse_3/Models/decoder_vae_19.h5')

vae = VariationalAutoencoder()

train_generator = vae.generate_keras_input('train')#val')

mse = []
ssim = []

for i in range(20):
    under, imgs = next(train_generator)
    imgs_true = imgs[0]
    
    params = encoder.predict(under)

    mu = params[:, :vae.latent_dim]
    log_var = params[:, vae.latent_dim:]

    z = mu + (np.random.normal(0,1,mu.shape) * np.exp(.5 * log_var))

    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))
    
val_generator = vae.generate_keras_input('val')#val')

mse = []
ssim = []

for i in range(20):
    under, imgs = next(val_generator)
    imgs_true = imgs[0]
    
    params = encoder.predict(under)

    mu = params[:, :vae.latent_dim]
    log_var = params[:, vae.latent_dim:]

    z = mu + (np.random.normal(0,1,mu.shape) * np.exp(.5 * log_var))

    imgs_rec = generator.predict(z)
    
    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))

print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))


from bigan import BiGAN, generate_keras_input

bi = BiGAN()

bi.load(5)

train_generator = generate_keras_input('train')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(train_generator)
    
    z = bi.GAN.E.predict(under, batch_size = 16)
    imgs_rec = bi.GAN.G.predict(z, batch_size = 16)

    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))
    
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

val_generator = generate_keras_input('val')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(val_generator)
    
    z = bi.GAN.E.predict(under, batch_size = 16)
    imgs_rec = bi.GAN.G.predict(z, batch_size = 16)

    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))
    
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim))) 

train_generator = generate_keras_input('train')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(train_generator)
    
    z = bi.GAN.EE.predict(under, batch_size = 16)
    imgs_rec = bi.GAN.GE.predict(z, batch_size = 16)

    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))
    
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

val_generator = generate_keras_input('val')

mse = []
ssim = []

for i in range(20):
    imgs_true, under, _ = next(val_generator)
    
    z = bi.GAN.EE.predict(under, batch_size = 16)
    imgs_rec = bi.GAN.GE.predict(z, batch_size = 16)

    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))
    
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))

for i in range(20):
    imgs_true, under, _ = next(val_generator)
    
    z = bi.GAN.EE.predict(under, batch_size = 16)
    imgs_rec = bi.GAN.GE.predict(z, batch_size = 16)

    mse.append(np.mean((imgs_true - imgs_rec)**2))

    imgs_true = K.constant(imgs_true)
    imgs_rec = K.constant(imgs_rec)

    ssim.append(K.eval(DSSIMObjective()(imgs_true, imgs_rec)))
    
print(np.mean(np.asarray(mse)))
print(np.mean(np.asarray(ssim)))