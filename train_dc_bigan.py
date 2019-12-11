import h5py
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from dc_bigan_3 import BIGAN

bigan = BIGAN()

epochs = 5

for epoch in tqdm(range(epochs)):
    input_generator = bigan.generate_keras_input('train')
    cntr = 0
    while(1):
        try:
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise and generate img            
            imgs, under, z = next(input_generator)
                        
            batch_size = imgs.shape[0]
            
            # flip labels 10% of the time
            flip = np.random.rand()
            
            if flip > 0:
                valid = np.ones((batch_size, 1))
                #fake = .05 * np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
            else:
                fake = np.ones((batch_size, 1))
                #valid = .05 * np.ones((batch_size, 1))
                valid = np.zeros((batch_size, 1))
            
            imgs_ = bigan.generator.predict(z)

            z_ = bigan.encoder.predict(under)

            # Train the discriminator (img -> z is valid, z -> img is fake)
            d_loss_real = bigan.discriminator.train_on_batch([z_, imgs], valid)
            d_loss_fake = bigan.discriminator.train_on_batch([z, imgs_], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (z -> img is valid and img -> z is is invalid)
            g_loss = bigan.bigan_generator.train_on_batch([z, under, imgs], [valid, fake])
            
            acc = d_loss[1]
            '''
            if acc > .65: #could be .65
                g_loss = bigan.bigan_generator.train_on_batch([z, imgs], [valid, fake])
                g_loss = bigan.bigan_generator.train_on_batch([z, imgs], [valid, fake])
                g_loss = bigan.bigan_generator.train_on_batch([z, imgs], [valid, fake])
                
                imgs_ = bigan.generator.predict(z)
                d_loss_real = bigan.discriminator.evaluate([z_, imgs], valid)
                d_loss_fake = bigan.discriminator.evaluate([z, imgs_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                acc = d_loss[-1]
                
            '''
            # Plot the progress
            print ("%d-%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, cntr, d_loss[0], 100*d_loss[-1], g_loss[0]))
            cntr = cntr + 1
        except Exception as e:
            print(e)
            
            batch_size = 20

            z = np.random.normal(size=(batch_size, bigan.latent_dim))
            imgs = bigan.generator.predict(z)

            for i in range(batch_size):
                mri = imgs[i, :, :, 0]

                plt.imshow(mri, cmap='gray')
                plt.savefig(f'bigan_3/Images/{epoch}_{i}.png')

            bigan.encoder.save(f'bigan_3/Models/encoder_{epoch}.h5')
            bigan.generator.save(f'bigan_3/Models/decoder_{epoch}.h5')    
            
            break
