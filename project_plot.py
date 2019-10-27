!ls
!git clone https://github.com/zsefbadgjl/514data.git 514data
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 22:58:34 2019

@author: HYH
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:11:40 2019

@author: HYH
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
#%matplotlib inline

from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K   # 'generic' backend so code works with either tensorflow or theano

K.clear_session()

np.random.seed(237)


#
train_orig = pd.read_csv('514data/train.csv')
test_orig = pd.read_csv('514data/test.csv')

train_orig.head()


# create 'label' column in test dataset; rearrange so that columns are in the same order as in train
test_orig['label'] = 11
testCols = test_orig.columns.tolist()
testCols = testCols[-1:] + testCols[:-1]
test_orig = test_orig[testCols]


# combine original train and test sets
combined = pd.concat([train_orig, test_orig], ignore_index = True)

combined.head()


combined.tail()




# Hold out 5000 random images as a validation/test sample
valid = combined.sample(n = 5000, random_state = 555)
train = combined.loc[~combined.index.isin(valid.index)]
plot100 = train_orig.sample(n = 100, random_state = 555)

# free up some space and delete test and combined
del train_orig, test_orig, combined

valid.head()






# X's
x_train = train.drop(['label'], axis = 1)
x_valid = valid.drop(['label'], axis = 1)
x_plot100 = plot100.drop(['label'], axis = 1)

# labels
y_train = train['label']
y_valid = valid['label']
y_plot100 = plot100['label']

# Normalize and reshape
x_train = x_train.astype('float32') / 255.
x_train = x_train.values.reshape(-1,28,28,1)

x_valid = x_valid.astype('float32') / 255.
x_valid = x_valid.values.reshape(-1,28,28,1)

x_plot100 = x_plot100.astype('float32') / 255.
x_plot100 = x_plot100.values.reshape(-1,28,28,1)






plt.figure(1)
plt.subplot(221)
plt.imshow(x_train[13][:,:,0])

plt.subplot(222)
plt.imshow(x_train[690][:,:,0])

plt.subplot(223)
plt.imshow(x_train[2375][:,:,0])

plt.subplot(224)
plt.imshow(x_train[42013][:,:,0])
plt.show()




#A. Encoder network
img_shape = (28, 28, 1)    # for MNIST
batch_size = 16
latent_dim = 10  # Number of latent dimension parameters

# Encoder architecture: Input -> Conv2D*4 -> Flatten -> Dense
input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', 
                  activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', 
                  activation='relu')(x)
# need to know the shape of the network here for the decoder
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Two outputs, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)




#B. Sampling function
# sampling function
def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

# sample vector from the latent distribution
z = layers.Lambda(sampling)([z_mu, z_log_sigma])



#C. Decoder network
# decoder takes the latent distribution sample as input
decoder_input = layers.Input(K.int_shape(z)[1:])

# Expand to 784 total pixels
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

# use Conv2DTranspose to reverse the conv layers from the encoder
x = layers.Conv2DTranspose(32, 3,
                           padding='same', 
                           activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(1, 3,
                  padding='same', 
                  activation='sigmoid')(x)

# decoder model statement
decoder = Model(decoder_input, x)

# apply the decoder to the sample from the latent distribution
z_decoded = decoder(z)




#D. Loss
# construct a custom layer to calculate the loss
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        # Reconstruction loss
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis=-1)
        return K.mean(xent_loss + kl_loss)

    # adds the custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

# apply the custom loss to the input images and the decoded latent distribution sample
y = CustomVariationalLayer()([input_img, z_decoded])



# VAE model statement
vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()



###4 Train the VAE
vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=7,
        batch_size=batch_size,
        validation_data=(x_valid, None))




###5
# Isolate original training set records in validation set
valid_noTest = valid[valid['label'] != 11]

# X's and Y's
x_valid_noTest = valid_noTest.drop('label', axis=1)
y_valid_noTest = valid_noTest['label']

# Reshape and normalize
x_valid_noTest = x_valid_noTest.astype('float32') / 255.
x_valid_noTest = x_valid_noTest.values.reshape(-1,28,28,1)




#!!!!!!!!*
plot100_noTest = plot100[plot100['label'] != 11]

# X's and Y's
x_plot100_noTest = plot100_noTest.drop('label', axis=1)
y_plot100_noTest = plot100_noTest['label']

# Reshape and normalize
x_plot100_noTest = x_plot100_noTest.astype('float32') / 255.
x_plot100_noTest = x_plot100_noTest.values.reshape(-1,28,28,1)

encoder = Model(input_img, z_mu)
x_plot100_noTest_encoded = encoder.predict(x_plot100_noTest, batch_size=batch_size)

def mtx_similar1(arr1:np.ndarray, arr2:np.ndarray) ->float:
   
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom 
    return  (similar+1) / 2
def mtx_similar2(arr1:np.ndarray, arr2:np.ndarray) ->float:
   
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    numera = np.sum(differ**2)
    denom = np.sum(arr1**2)
    similar = 1 - (numera / denom)
    if similar <0:
        return 0
    else:    
        return similar
def mtx_similar3(arr1:np.ndarray, arr2:np.ndarray) ->float:
   
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    if similar <0:
        return 0
    else:    
        return similar
def mtx_similar(arr1:np.ndarray, arr2:np.ndarray) ->float:
    similar1 = mtx_similar1(arr1,arr2)
    similar2 = mtx_similar2(arr1,arr2)
    similar3 = mtx_similar3(arr1,arr2)
    if similar1 > 0.9:
        similar =  7/9*similar1 + 2/9*similar2
    if similar1 < 0.8:
        similar =  1/6*similar1 + 2.5/6*similar2 + 2.5/6*similar3
    if similar1 >=0.8 and similar1 <=0.9:
        similar =  4/9*similar1 + 1/3*similar2 + 2/9*similar3
    return similar







#xvplot
#plot the predicted image of x by decoding the encoded x

#loop from fi = 0 to 100
average = 0
for fi in range(0,100):
    digit_size = 28
    figure = np.zeros((digit_size * 1, digit_size * 1))
#    fi = 95
    #x_valid_noTest_encoded
    nparry = []
    for i in range(0,latent_dim):
      nparry.append(x_plot100_noTest_encoded[fi][i])
    x_decoded_plot100 = decoder.predict(np.array([nparry]), batch_size=batch_size)
    digit_predicted = x_decoded_plot100[0].reshape(digit_size, digit_size)
    figure[0: digit_size, 0: digit_size] = digit_predicted
#    plt.figure(figsize=(10, 10))
#    plt.imshow(figure, cmap='gnuplot2')
#    plt.show()  
    img1 = figure
    
    #plot the original x
    digit_size = 28
    figure = np.zeros((digit_size * 1, digit_size * 1))
    
    digit_orig = x_plot100_noTest[fi].reshape(digit_size, digit_size)
    figure[0: digit_size, 0: digit_size] = digit_orig
    
#    plt.figure(figsize=(10, 10))
#    plt.imshow(figure, cmap='gnuplot2')
#    plt.show()  
    img2 = figure
    

#    print (fi)
    n1 = mtx_similar1(img1,img2)
#    print (n1)
    n2 = mtx_similar2(img1,img2)
#    print (n2)
    n3 = mtx_similar3(img1,img2)
#    print (n3)
    n = mtx_similar(img1,img2)
#    print (n)
    average = average + n
average = average / 100
print (average)

#plot generated image
figure = np.zeros((digit_size * 1, digit_size * 1))

nparry = []
for i in range(0,latent_dim):
  nparry.append(x_plot100_noTest_encoded[24][i])
x_decoded_plot100 = decoder.predict(np.array([nparry]), batch_size=batch_size)
digit_predicted = x_decoded_plot100[0].reshape(digit_size, digit_size)
figure[0: digit_size, 0: digit_size] = digit_predicted
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show()

#plot the original x
digit_size = 28
figure = np.zeros((digit_size * 1, digit_size * 1))
    
digit_orig = x_plot100_noTest[24].reshape(digit_size, digit_size)
figure[0: digit_size, 0: digit_size] = digit_orig
    
plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gnuplot2')
plt.show() 
    






####*!!!!!!!!!!!!!!!!!





# # Translate into the latent space
# encoder = Model(input_img, z_mu)
# x_valid_noTest_encoded = encoder.predict(x_valid_noTest, batch_size=batch_size)
# plt.figure(figsize=(10, 10))
# plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1], c=y_valid_noTest, cmap='brg')
# plt.colorbar()
# plt.show()



# # set colormap so that 11's are gray
# custom_cmap = matplotlib.cm.get_cmap('brg')
# custom_cmap.set_over('gray')

# x_valid_encoded = encoder.predict(x_valid, batch_size=batch_size)
# plt.figure(figsize=(10, 10))
# gray_marker = mpatches.Circle(4,radius=0.1,color='gray', label='Test')
# plt.legend(handles=[gray_marker], loc = 'best')
# plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c=y_valid, cmap=custom_cmap)
# plt.clim(0, 9)
# plt.colorbar()




# # Display a 2D manifold of the digits
# n = 20  # figure with 20x20 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))

# # Construct grid of latent variable values
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

# # decode for each square in the grid
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
#         x_decoded = decoder.predict(z_sample, batch_size=batch_size)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[i * digit_size: (i + 1) * digit_size,
#                j * digit_size: (j + 1) * digit_size] = digit

# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap='gnuplot2')
# plt.show()  








#import sys
#
#import tensorflow.keras
#import pandas as pd
#import sklearn as sk
#import tensorflow as tf
#
#print(f"Tensor Flow Version: {tf.__version__}")
#print(f"Keras Version: {tensorflow.keras.__version__}")
#print()
#print(f"Python {sys.version}")
#print(f"Pandas {pd.__version__}")
#print(f"Scikit-Learn {sk.__version__}")
#print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
#


