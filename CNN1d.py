from tensorflow.keras.models import Sequential
import glob
from tensorflow.keras.layers import Dense,GlobalAveragePooling1D,Conv2D,MaxPool2D,Flatten,Input,Dropout,Conv1D,BatchNormalization,MaxPool1D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from useful_class import Preprocessing,Painting
from sklearn.decomposition import PCA
import tensorflow as tf
files = glob.glob(r'D:\Solar panels\average\FX10\*.npy')

gt = []
all_samples =[]
for i, f in enumerate(files):
    current_file = np.load(f)
    for cf in current_file:
        sample = cf[5:215]
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


#%%
from scipy.signal import savgol_filter
all_samples_snv = Preprocessing.snv(np.array(all_samples))
all_samples_sf = savgol_filter(all_samples_snv,13,5)
components = 20
pca = PCA(components)
all_samples_pca = pca.fit_transform(all_samples_snv)

#%%
bands = 210
data = np.array(all_samples_sf).reshape(-1,bands,1)
x,y,z = data.shape
state = None
gt = to_categorical(gt,4)
train_hsi_samples,test_hsi_samples,train_gt,test_gt= train_test_split(data,gt,test_size=0.7,random_state=state)

model = Sequential()
model.add(Conv1D(32, 3, input_shape=(bands,1), padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(Conv1D(16, 3, padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(MaxPool1D())
model.add(Conv1D(32, 3, input_shape=(bands,1), padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(Conv1D(16, 3, padding="same", activation="relu",strides=1))
model.add(BatchNormalization())
model.add(MaxPool1D())

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(4, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
checkpoint_filepath = "D:\Solar panels\model\cnn1d.h5"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)


history = model.fit(train_hsi_samples,train_gt,epochs=1000,batch_size=2,
                    validation_split=0.2,verbose=1,
                    shuffle=True,callbacks=[model_checkpoint_callback,reduce_lr,es])

# model.save(r"D:\Solar panels\model\cnn1d.h5")
model.load_weights(checkpoint_filepath)
result = model.predict(test_hsi_samples)
acc = accuracy_score(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
cm = confusion_matrix(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
kappa = cohen_kappa_score(np.argmax(test_gt,axis=1),np.argmax(result,axis=1))
types = ('chip_b','chip_w','line','glass')
Painting.plot_confusion_matrix(cm, types)

