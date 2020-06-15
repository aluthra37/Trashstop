from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from zipfile import ZipFile
from PIL import Image
import os
import shutil
import glob

# Change directory to the zipfile annd extractingn the files
os.chdir('/Users/ruthvikpedibhotla/Documents/Python')
with ZipFile('garbage_photos.zip','r') as garbage_photos:
    garbage_photos.extractall()

# Removing the useless directory that shows up after unzipping
shutil.rmtree('/Users/ruthvikpedibhotla/Documents/Python/__MACOSX')
PATH=os.path.join(os.path.dirname('garbage_photos.zip'),'/Users/ruthvikpedibhotla/Documents/Python')
trainDir=os.path.join(PATH,'garbage_photos')

# Changing directories again to count the total images and assign training directories
trainCardboard=os.path.join(trainDir,'cardboard')
trainGlass=os.path.join(trainDir,'glass')
trainMetal=os.path.join(trainDir,'metal')
trainPaper=os.path.join(trainDir,'paper')
trainPlastic=os.path.join(trainDir,'plastic')
trainTrash=os.path.join(trainDir,'trash')

# Counting the total images
imageCount=len(os.listdir(trainCardboard))+len(os.listdir(trainGlass))+len(os.listdir(trainMetal))+len(os.listdir(trainPaper))+len(os.listdir(trainPlastic))+len(os.listdir(trainTrash))
print('Total number of Training Images: ',imageCount)

# Setting Up Variables
batchSize=16
epochs=20
imgH=64
imgW=64

# Reading Images from the Disk and Preparing
trainImageGenerator=ImageDataGenerator(rescale=1./255)
trainDataGen=trainImageGenerator.flow_from_directory(batch_size=batchSize,
                                                           directory=trainDir,
                                                           shuffle=True,
                                                           target_size=(imgH, imgW),
                                                           class_mode='categorical')

# Visualize Training Image
sampleTrainImages,_=next(trainDataGen)

# Plot Images in Grid Form
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Creating the Model
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(imgH, imgW ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(6, activation='softmax')])

# Compile the Model loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
history=model.fit(trainDataGen,
    steps_per_epoch=imageCount//batchSize,
    epochs=epochs)

# Creating Model
model.save('model.h5')

# Visualize the model
# Validation Accuracy
acc=history.history['accuracy']
# Validation Loss
loss=history.history['loss']

epochsRange = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochsRange, acc, label='Training Accuracy')
# Training Accuracy
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochsRange, loss, label='Training Loss')
# Training Loss
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
