import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from useful_class import Painting
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint ,ReduceLROnPlateau ,EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (GlobalAveragePooling2D,MaxPooling2D,Activation,
                                     Dense,GlobalAveragePooling1D,
                                     Conv2D,GlobalMaxPool1D,Flatten,
                                     Input,Dropout,Conv1D,BatchNormalization,MaxPool1D)


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)
    
def show_train_history(train_history):
    plt.figure()
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train acc", "val acc"], loc="best")
#    plt.figure()
    plt.subplot(122)    
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.legend(["train loss", "val loss"], loc="best")
    plt.show()
#%% Data


files = glob.glob(r'D:\Solar panels\roi\*.npy')


gt = []
all_samples =[]
for i, f in enumerate(files):
    current_file = np.load(f)
    for cf in current_file:
        sample = cf
        all_samples.append(sample)
        f = f.split('\\')[-1]
        if 'b' in f:
            gt.append(0)
        elif 'w' in f:
            gt.append(1)
        elif 'l' in f:
            gt.append(2)
        elif 'g' in f:
            gt.append(3)

gt = to_categorical(gt,4)
all_samples = np.array(all_samples)
train_hsi_samples,test_hsi_samples,train_gt,test_gt = train_test_split(all_samples,gt,test_size=0.5,random_state=789456)
datagen = ImageDataGenerator(rotation_range=15 ,width_shift_range=0.2 ,height_shift_range=0.2 ,
                             zoom_range=0.2 ,validation_split=0.3,data_format='channels_last')


#%%


model = Sequential()
model.add(Conv2D(32,(3,3),strides=(1,1),input_shape=(87,185,224),padding='same'))  
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same'))
model.add(Activation('relu'))
#model.add(MaxPooling2D())
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same'))
#model.add(Activation('relu'))


model.add(GlobalAveragePooling2D())  
model.add(Dense(128,activation='relu'))    
model.add(Dense(4,activation='softmax'))


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc']) 
model.summary()

h5_path = r"D:\Solar panels\model\2dcnn"
es = EarlyStopping(patience=15, verbose=1)
checkpoint = ModelCheckpoint(h5_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',verbose = 1, factor=0.5, patience=5, min_lr=1e-6, mode = 'auto')

batch_size = 5
epochs = 100

datagen.fit(train_hsi_samples,augment=True)
history =model.fit(
    datagen.flow(train_hsi_samples, train_gt, 
                 batch_size=batch_size,subset='training'),
    validation_data = datagen.flow(train_hsi_samples, train_gt,
                                 batch_size=5, subset='validation'),
    batch_size = batch_size,
    steps_per_epoch = len(train_hsi_samples)//batch_size,
    validation_steps=len(train_hsi_samples)//batch_size,
    epochs = epochs,
    callbacks=[reduce_lr,checkpoint,es])


# history = model.fit(train_hsi_samples,
#                     train_gt,
#                     batch_size=2,
#                     epochs=100,
#                     shuffle=True,
#                     verbose=1,
#                     validation_split=0.3,
#                     callbacks=[reduce_lr,checkpoint,es])


show_train_history(history)

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
result = model.predict(test_hsi_samples)
acc = accuracy_score(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
cm = confusion_matrix(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
kappa = cohen_kappa_score(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
types = ('chip_b','chip_w','line','glass')
Painting.plot_confusion_matrix(cm, types)





    