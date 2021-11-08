# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:21:45 2021

@author: user
"""

import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation,measure,morphology,color

def LDA_Model(X_train,y_train):
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X_train, y_train) 
    return model



path=r"D:\Solar panels\sun_20211014\3"
img =envi.open(path+'_RT.hdr',path+'_RT.raw')
img =img.open_memmap(writeable=True)
w,h,b = img.shape
img_1d = img.reshape(-1,b)

plt.figure()
plt.imshow(img[:,:,100])
points = plt.ginput(20)
points = np.array(points).astype(int)
plt.close()


gt= [3]*5 + [2]*5 + [1]*5 + [0]*5
train_data = []
for point in points :
    train_data.append(img[point[1],point[0],:])

model = LDA_Model(train_data,gt)
ans = model.predict(img_1d)
plt.imshow(ans.reshape(w,h))

