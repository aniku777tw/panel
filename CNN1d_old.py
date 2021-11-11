# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 00:10:26 2021

@author: HSIPL39
"""


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Activation
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D,Conv2D,MaxPool2D,Flatten,Input,Dropout,Conv1D,BatchNormalization,MaxPool1D
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from hsipl_tools import target_detection as td
from hsipl_tools import anomaly_detection as ad
from hsipl_tools import preprocessing as pp
from hsipl_tools import band_selection as bs
from skimage import segmentation,measure,filters
from spectral import envi
import spectral
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score,confusion_matrix,cohen_kappa_score
from keras.utils import to_categorical
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FastICA




grape_average = np.load('grape_average.npy')




groundtruth=[15.5,15.5,15,16,15,16.5,15,16,16,15.5,16.5,15,15,15,16,16.5]
gt = []
for num in groundtruth:
    if num > 15.5:
        gt.append(1)
    else :
        gt.append(0)
plt.figure()    
for i in range(0,16):
    if gt[i]==1:
        plt.plot(range(1,193),grape_average[i],'b') 
    else :
        plt.plot(range(1,193),grape_average[i],'r') 
        

data = np.array(grape_average).reshape(-1,192,1)
#bands_data = np.array(bands_data)
#bands = bs.CEM_BDM(bands_data,5)
#new_data = []
#for band in bands:
#    new_data.append(data[:,band])
#new_data = np.array(new_data).reshape(379,5,1)
x,y,z = data.shape
#PCA = PCA(components)
#fastica = FastICA(components)
#spca = SparsePCA(components)
#data = spca.fit_transform(data.reshape(-1,224)).reshape(x,components,1)
#data = PCA.fit_transform(data.reshape(-1,224)).reshape(x,components,1)
#data = fastica.fit_transform(data.reshape(-1,224)).reshape(x,components,1)
gt = to_categorical(gt,2)
x_train,y_train,x_test,y_test = train_test_split(data,gt,test_size=0.5, random_state=7878)
model = Sequential()
#
model.add(Conv1D(32, 3, input_shape=(192,1), padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(Conv1D(16, 3, padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(MaxPool1D())
model.add(Conv1D(32, 3, input_shape=(192,1), padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(Conv1D(16, 3, padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(MaxPool1D())

#model.add(Conv1D(8, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(64, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(MaxPool1D())
#model.add(Conv1D(128, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(128, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(128, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(MaxPool1D())
#model.add(Conv1D(256, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(256, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(256, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(MaxPool1D())
#model.add(Conv1D(512, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(512, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(512, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(MaxPool1D())
#model.add(Conv1D(1024, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(1024, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(Conv1D(1024, 3, padding="same", activation="relu",strides=1))
#model.add(BatchNormalization())
#model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train,x_test,epochs=10,batch_size=32,validation_split=0.2,verbose=1,shuffle=True)
model.save(r"D:\fruit\model\cnn1d.h5")
result = model.predict(y_train)
acc = accuracy_score(np.argmax(y_test,axis=1),np.argmax(result,axis=1))
cm = confusion_matrix(np.argmax(y_test,axis=1),np.argmax(result,axis=1))
kappa = cohen_kappa_score(np.argmax(y_test,axis=1),np.argmax(result,axis=1))
