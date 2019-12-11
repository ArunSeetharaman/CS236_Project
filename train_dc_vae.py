import h5py
import numpy as np
from matplotlib import pyplot as plt

from dc_vae_3 import VariationalAutoencoder

vae = VariationalAutoencoder()

input_generator = vae.generate_keras_input('train')
val_generator = vae.generate_keras_input('val')

epochs = 20

for epoch in range(epochs):
    vae.variational_autoencoder.fit_generator(input_generator, 973, 1, validation_data=val_generator, validation_steps=199)

    vae.encoder.save(f'vae_mse_3/Models/encoder_vae_{epoch}.h5')
    vae.decoder.save(f'vae_mse_3/Models/decoder_vae_{epoch}.h5')

    batch_size = 20

    z = np.random.normal(size=(batch_size, vae.latent_dim))
    imgs = vae.decoder.predict(z)

    for i in range(batch_size):
        mri = imgs[i, :, :, 0]

        plt.imshow(mri, cmap='gray')
        plt.savefig(f'vae_mse_3/Images/{epoch}_{i}_vae.png')
