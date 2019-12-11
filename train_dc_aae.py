import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dc_aae_2 import AdversarialAutoencoder

aae =  AdversarialAutoencoder()

epochs = 25#20

for epoch in tqdm(range(epochs)):
    input_generator = aae.generate_keras_input('train')
    cntr = 0
    while(1):
        try:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random batch of images
            imgs, latent_real = next(input_generator)

            latent_fake = aae.encoder.predict(imgs)
            
            batch_size = imgs.shape[0]
            
            # Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = aae.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = aae.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = aae.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print ("%d-%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, cntr, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            cntr = cntr + 1
            
        except Exception as e:
            print(e)
            batch_size = 20

            z = np.random.normal(size=(batch_size, aae.latent_dim))
            imgs = aae.decoder.predict(z)

            for i in range(batch_size):
                mri = imgs[i, :, :, 0]

                plt.imshow(mri, cmap='gray')
                plt.savefig(f'aae_huber_ssim_2/Images/{epoch}_{i}_aae_ssim.png')
            aae.encoder.save(f'aae_huber_ssim_2/Models/encoder_aae_ssim_{epoch}.h5')
            aae.decoder.save(f'aae_huber_ssim_2/Models/decoder_aae_ssim_{epoch}.h5')    
            
            break
            
