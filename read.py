from spectral.io import envi
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from skimage import segmentation,measure,morphology,color
from hsipl_tools.preprocessing import snv

def roi(img,mask,area_value):
    try: 
        roi = []
        cleared = mask.copy() 
        segmentation.clear_border(cleared)
        label_image =measure.label(cleared)  
        borders = np.logical_xor(mask, cleared) 
        label_image[borders] = -1
        x = 2
        for region in measure.regionprops(label_image):      
        
            if region.area <area_value:
                continue
            print(np.array(region.bbox))
            minr, minc, maxr, maxc = region.bbox      
            roi.append(img[minr-x:maxr+x,minc-x:maxc+x,:])
    except:
        print('ROI error')
    else:
        return roi

def LDA_Model(X_train,y_train):
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X_train, y_train) 
    return model

def SVM_Model(X_train,y_train):
    
    from sklearn.svm import SVC
    svm = SVC()
    model = svm.fit(X_train, y_train) 
    return model

def SaveModel(model,tag):
    import pickle
    with open('model/' + tag + '_noback.pkl', 'wb') as f:
        pickle.dump(model, f)
        
def LoadModel(tag):
    import pickle
    with open('model/' + tag + '_noback.pkl', 'rb') as f:
        model = pickle.load(f)
    return model
def masking(img,mask):
    new_img = img.copy()
    for i in range(img.shape[2]):
        new_img[...,i] = mask*img[...,i]
    return new_img



data_name = 'l11'
data_path = r"D:\Solar panels\20211105\\" +data_name +"_RT"
img = envi.open(data_path + '.hdr',data_path + '.raw').asarray()
w,h,b = img.shape
# img_1d = img.reshape(-1,b)

#%% train masking model

# plt.imshow(img[:,:,101])
# points = plt.ginput(25,timeout=120)
# points = np.array(points).astype(int)
# plt.close()
# gt=[1]*20 + [0]*5
# train_data = []
# for point in points :
#     train_data.append(img[point[1],point[0],:])

# # img_1d = snv(img_1d)

# model = SVM_Model(train_data,gt)
# ans = model.predict(img_1d)
# plt.imshow(ans.reshape(w,h))
# SaveModel(model,'remove_bg_svm')


#%% roi & masking

# rb_model = LoadModel('remove_bg_svm')
# mask = rb_model.predict(img_1d).reshape(w,h)
# img_masked = masking(img,mask)

# roi_img=roi(img_masked,mask,400)
# # plt.figure()
# # plt.imshow(mask)
# # plt.figure()
# # plt.imshow(img_masked[:,:,101])
# # plt.figure()
# for i in range(1,len(roi_img)+1):
#     try:
#         plt.subplot(10, 10, i)
#         plt.imshow(roi_img[i-1][...,100])
#     except:
#         pass


#%% 平均光譜



# hsi_mean = []
# for i in range(len(roi_img)):
#     hsi = roi_img[i]
#     hsi_1d = hsi.reshape(-1,b)
#     hsi_1d[hsi_1d == 0] = np.nan
#     mean = np.nanmean(hsi_1d,axis=0)
#     hsi_mean.append(mean)


# plt.figure()    
# for i in range(len(roi_img)):
#     plt.plot(range(b),hsi_mean[i])
# np.save('average/FX10/'+data_name+'_average',hsi_mean)



#%% label 
import glob

files = glob.glob(r'D:\Solar panels\average\FX10\*.npy') 

average_hsi_sample = []
gt = []
for i,f in enumerate(files) :
    current_file = np.load(f)
    for cf in current_file:
        average_hsi_sample.append(cf)
        f = f.split('\\')[-1]
        if 'b' in  f:
            gt.append(0)
        elif 'w' in  f:
            gt.append(1)
        elif 'l' in  f:
            gt.append(2)
        elif 'g' in  f:
            gt.append(3)

#%%
# 光譜前處理SNV    
snv_ahs= average_hsi_sample
# ahs = np.array(average_hsi_sample)
# snv_ahs = snv(ahs)

# 平均光譜圖

plt.figure()    
for i in range(len(gt)):
    if  gt[i]==0:
        plt.plot(range(b),snv_ahs[i],'b') 
    elif gt[i]==1:
        plt.plot(range(b),snv_ahs[i],'r') 
    elif gt[i]==2:
        plt.plot(range(b),snv_ahs[i],'g') 
    elif gt[i]==3:
        plt.plot(range(b),snv_ahs[i],'y') 
# PCA 降維        
pca = PCA(3)
pca_res = pca.fit_transform(snv_ahs)
evr = sum(pca.explained_variance_ratio_)
ev = pca.explained_variance_


# SVM model 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_smaple,test_smaple,train_gt,test_gt = train_test_split(pca_res,gt,test_size=0.5)
svm = SVC()
model = svm.fit(train_smaple,train_gt)

# predict result

result = model.predict(test_smaple)
acc = accuracy_score(test_gt,result)
stack = np.stack((result,test_gt),axis=1)

