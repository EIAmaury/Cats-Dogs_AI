# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:03:24 2024

@author: amaur
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Normalization
from PIL import Image

sample_example=4
fig,ax=plt.subplots(1,sample_example)
for i in range(sample_example):
    # Chemin relatif des images d'entrainement
    img = Image.open('train\\train\\images\\dog\\dog_image{}.jpg'.format(12500+i))
    sample_case = img.resize((200,200))
    img_array = np.array(sample_case)
    ax[i].imshow(img_array)
    ax[i].set_title("original")
    ax[i].axis("off")
plt.show()

fig,ax=plt.subplots(1,sample_example)
for i in range(sample_example):
    # Chemin relatif des images d'entrainement
    img = Image.open('train\\train\\images\\cat\\cat_image{}.jpg'.format(i))
    sample_case = img.resize((200,200))
    img_array = np.array(sample_case)
    ax[i].imshow(img_array)
    ax[i].set_title("original")
    ax[i].axis("off")
plt.show()


#%% model 
train_ds = keras.utils.image_dataset_from_directory(
    directory='train\\train\\images\\',# Chemin relatif des images
    labels='inferred',
    class_names=['cat','dog'],
    label_mode='categorical',
    batch_size=100,
    image_size=(170, 170))
#%%
model=Sequential([
    Normalization(input_shape=(170, 170, 3), name='normalization'),
    Conv2D(128,(3,3),activation='relu',padding='valid'),
    MaxPooling2D((5,5)),
    Conv2D(256,(3,3),activation='relu',padding='valid'),
    MaxPooling2D((5, 5)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(2,activation='softmax')
    ])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ds,epochs=2)

#%% test of data
from PIL import Image
sample_example=7

fig,ax=plt.subplots(1,sample_example)
for i in range(sample_example):
    # Chemin relatif des images de test
    img = Image.open('test1\\test1\\{}.jpg'.format(1000+i))
    sample_case = img.resize((170,170))
    img_array = np.array(sample_case)
    ax[i].imshow(img_array)
    ax[i].set_title("original")
    ax[i].axis("off")
    x_image_test=tf.convert_to_tensor(img_array)
    x_image_test=tf.reshape(x_image_test,[-1,170,170,3])
    x_image_test=tf.cast(x_image_test,'float32')
    predictions=model.predict(x_image_test)
    number=tf.argmax(predictions,axis=-1)
    #l sil me sort un numéro il faut le faire correspondre à la classe if==0 and if ==1
    if number==0:
        print("This image is a cat")
    else:
        print("This image is a dog")
plt.show()
print("----------------")
sample_example=7

fig,ax=plt.subplots(1,sample_example)
for i in range(sample_example):
    # Chemin relatif des images de test
    img = Image.open('test1\\test1\\{}.jpg'.format(1007+i))
    sample_case = img.resize((170,170))
    img_array = np.array(sample_case)
    ax[i].imshow(img_array)
    ax[i].set_title("original")
    ax[i].axis("off")
    x_image_test=tf.convert_to_tensor(img_array)
    x_image_test=tf.reshape(x_image_test,[-1,170,170,3])
    x_image_test=tf.cast(x_image_test,'float32')
    predictions=model.predict(x_image_test)
    number=tf.argmax(predictions,axis=-1)
    #l sil me sort un numéro il faut le faire correspondre à la classe if==0 and if ==1
    if number==0:
        print("This image is a cat")
    else:
        print("This image is a dog")
plt.show()




    