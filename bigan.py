# This is the moving average BiGAN for image domain.

from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial
from random import random
import os
import h5py
import cv2
import matplotlib.pyplot as plt

im_size = 160
latent_size = 64
BATCH_SIZE = 16
directory = "train"
suff = 'png'
cmode = 'L'
channels = 1
size_adjusted = False

k_images = 3

cha = 16

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size])

def generate_keras_input(mode = 'train'):
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
    while(1):
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

            z = np.random.normal(size=(batch_size, latent_size))

            yield [x, under, z]    

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r %s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
        print()

from keras.layers import Conv2D, Dense, AveragePooling2D, Activation, Cropping2D, Dropout, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Flatten, Input, add, Lambda, concatenate, LeakyReLU, multiply
from keras.layers import GlobalAveragePooling2D, average
from keras.models import model_from_json, Model
from keras.initializers import VarianceScaling
from keras.optimizers import Adam
import keras.backend as K

def gradient_penalty_loss(y_true, y_pred, averaged_samples, sample_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * (sample_weight / 2)

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 - (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

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

class GAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):

        #Models
        self.D = None
        self.E = None
        self.G = None

        self.GE = None
        self.EE = None

        self.DM = None
        self.AM = None

        #Config
        self.LR = lr
        self.steps = steps
        self.beta = 0.999

        #Init Models
        self.discriminator()
        self.generator()
        self.encoder()

        self.EE = model_from_json(self.E.to_json())
        self.EE.set_weights(self.E.get_weights())

        self.GE = model_from_json(self.G.to_json())
        self.GE.set_weights(self.G.get_weights())

    def discriminator(self):

        if self.D:
            return self.D

        inp = Input(shape = [im_size, im_size, 1])
        inpl = Input(shape = [latent_size])

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

        x = Dense(1, kernel_initializer = 'he_normal')(x)

        self.D = Model(inputs = [inp, inpl], outputs = x)

        return self.D

    def generator(self):

        if self.G:
            return self.G

        #Inputs
        inp = Input(shape = [latent_size])

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

        self.G = Model(inputs = inp, outputs = x)

        return self.G

    def encoder(self):

        if self.E:
            return self.E

        inp = Input(shape = [im_size, im_size, 1])

        x = d_block(inp, 1 * cha)   #64
        x = d_block(x, 2 * cha)   #32
        x = d_block(x, 3 * cha)   #16
        x = d_block(x, 4 * cha)  #8
        x = d_block(x, 8 * cha)  #4
        x = d_block(x, 16 * cha, p = False)  #4

        x = Flatten()(x)

        x = Dense(16 * cha, kernel_initializer = 'he_normal')(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(latent_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)

        self.E = Model(inputs = inp, outputs = x)
        
        return self.E

    def AdModel(self):

        #D does not update
        self.D.trainable = False
        for layer in self.D.layers:
            layer.trainable = False

        #G does update
        self.G.trainable = True
        for layer in self.G.layers:
            layer.trainable = True

        #E does update
        self.E.trainable = True
        for layer in self.E.layers:
            layer.trainable = True

        # Fake Latent / Real Image
        ri = Input(shape = [im_size, im_size, 1])
        
        ui = Input(shape = [im_size, im_size, 1])

        er = self.E(ui)
        dr = self.D([ri, er])

        # Real Latent / Fake Image
        gi = Input(shape = [latent_size])

        gf = self.G(gi)
        df = self.D([gf, gi])

        self.AM = Model(inputs = [ri, ui, gi], outputs = [dr, df])

        self.AM.compile(optimizer = Adam(self.LR, beta_1 = 0, beta_2 = 0.099), loss = [w_loss, w_loss])

        return self.AM

    def DisModel(self):

        #D does update
        self.D.trainable = True
        for layer in self.D.layers:
            layer.trainable = True

        #G does not update
        self.G.trainable = False
        for layer in self.G.layers:
            layer.trainable = False

        #E does update
        self.E.trainable = False
        for layer in self.E.layers:
            layer.trainable = False

        # Fake Latent / Real Image
        ri = Input(shape = [im_size, im_size, 1])
        
        ui = Input(shape = [im_size, im_size, 1])

        er = self.E(ui)
        dr = self.D([ri, er])

        # Real Latent / Fake Image
        gi = Input(shape = [latent_size])

        gf = self.G(gi)
        df = self.D([gf, gi])

        self.DM = Model(inputs = [ri, ui, gi], outputs = [dr, df, df])

        # Create partial of gradient penalty loss
        # For r1, averaged_samples = ri
        # For r2, averaged_samples = gf
        # Weight of 10 typically works
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = [gf, gi], sample_weight = 5)

        #Compile With Corresponding Loss Functions
        self.DM.compile(optimizer = Adam(self.LR, beta_1 = 0, beta_2 = 0.909), loss=[hinge_d, hinge_d, partial_gp_loss])

        return self.DM

    def EMA(self):

        start = time.clock()

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.E.layers)):
            up_weight = self.E.layers[i].get_weights()
            old_weight = self.EE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.EE.layers[i].set_weights(new_weight)

        #print("Moved Average. " + str(time.clock() - start) + "s")

    def MAinit(self):
        self.EE.set_weights(self.E.get_weights())
        self.GE.set_weights(self.G.get_weights())






class BiGAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):

        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()

        self.lastblip = time.clock()

        self.noise_level = 0

        #self.im = dataGenerator(directory, suffix = suff, flip = False)

        self.silent = silent

        #Train Generator to be in the middle, not all the way at real. Apparently works better??
        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones
        
        self.train_data = generate_keras_input(mode = 'train')
        self.test_data = generate_keras_input(mode = 'val')

    def train(self):

        #Train Alternating
        a = self.train_dis()
        b = self.train_gen()

        if self.GAN.steps % 10 == 0:
            self.GAN.EMA()

        if self.GAN.steps == 20000:
            self.GAN.MAinit()


        #Print info
        if self.GAN.steps % 100 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D: " + str(a))
            print("G: " + str(b))
            s = round((time.clock() - self.lastblip), 4)
            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))
            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            self.lastblip = time.clock()
            steps_left = 200000 - self.GAN.steps + 1e-7
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 1000):
                self.evaluate(floor(self.GAN.steps / 1000))


        printProgressBar(self.GAN.steps % 100, 99, decimals = 0)

        self.GAN.steps = self.GAN.steps + 1

    def train_dis(self):

        #Get Data
        train_data = next(self.train_data)
        #self.im.get_batch(BATCH_SIZE)#[self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE)]

        #Train
        d_loss = self.DisModel.train_on_batch(train_data, [self.ones, self.nones, self.ones])

        return d_loss

    def train_gen(self):

        #Train
        train_data = next(self.train_data)
        #self.im.get_batch(BATCH_SIZE)#[self.im.get_batch(BATCH_SIZE), noise(BATCH_SIZE)]

        g_loss = self.AdModel.train_on_batch(train_data, [self.ones, self.nones])

        return g_loss

    def evaluate(self, num = 0):

        n1 = noise(32)

        generated_images = self.GAN.G.predict(n1, batch_size = BATCH_SIZE)

        data = next(self.test_data)
        real_images = data[0]
        under = data[1]
                
        latent_codes = self.GAN.E.predict(under, batch_size = BATCH_SIZE)
        reconstructed_images = self.GAN.G.predict(latent_codes, batch_size = BATCH_SIZE)
                
        print("E Mean: " + str(np.mean(latent_codes)))
        print("E Std: " + str(np.std(latent_codes)))
        print("E Std Featurewise: " + str(np.mean(np.std(latent_codes, axis = 0))))
        print()
        
        r = []

        for i in range(0, 32, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        hline = np.zeros([16, 8 * im_size, 1])
        r.append(hline)

        for i in range(0, 16, 8):
            r.append(np.concatenate(real_images[i:i+8], axis = 1))
            r.append(np.concatenate(reconstructed_images[i:i+8], axis = 1))          

        c1 = np.concatenate(r, axis = 0)[:,:,0]
        
        plt.imshow(c1, cmap='gray')
        plt.savefig("BiGAN/Results/i"+str(num)+".png")

        #x = Image.fromarray(np.uint8(c1*255))

        #x.save("Results/i"+str(num)+".png")

        # Moving Average

        n1 = noise(32)

        generated_images = self.GAN.GE.predict(n1, batch_size = BATCH_SIZE)

        latent_codes = self.GAN.EE.predict(under, batch_size = BATCH_SIZE)
        reconstructed_images = self.GAN.GE.predict(latent_codes, batch_size = BATCH_SIZE)

        r = []

        for i in range(0, 32, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        hline = np.zeros([16, 8 * im_size, 1])
        r.append(hline)

        for i in range(0, 16, 8):
            r.append(np.concatenate(real_images[i:i+8], axis = 1))
            r.append(np.concatenate(reconstructed_images[i:i+8], axis = 1))
        
        c1 = np.concatenate(r, axis = 0)[:,:,0]
        
        plt.imshow(c1, cmap='gray')
        plt.savefig("BiGAN/Results/i"+str(num)+"-ema.png")
                    
        #x = Image.fromarray(np.uint8(c1*255))

        #x.save("Results/i"+str(num)+"-ema.png")

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("BiGAN/Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("BiGAN/Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):

        file = open("BiGAN/Models/"+name+".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json)
        mod.load_weights("BiGAN/Models/"+name+"_"+str(num)+".h5")

        return mod

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)
        self.saveModel(self.GAN.E, "enc", num)

        self.saveModel(self.GAN.GE, "genMA", num)
        self.saveModel(self.GAN.EE, "encMA", num)


    def load(self, num): #Load JSON and Weights from /Models/
        steps1 = self.GAN.steps

        #Load Models
        self.GAN.G = self.loadModel("gen", num)
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.E = self.loadModel("enc", num)

        self.GAN.GE = self.loadModel("genMA", num)
        self.GAN.EE = self.loadModel("encMA", num)

        self.GAN.steps = steps1

        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()




if __name__ == "__main__":
    model = BiGAN(lr = 0.0001, silent = False)
    model.evaluate(0)

    while model.GAN.steps <= 50000:#600000:
        model.train()
