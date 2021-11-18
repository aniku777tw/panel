import glob
from spectral.io import envi
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from hsipl_tools.preprocessing import snv,msc,data_normalize
from useful_class import ModelAccess,Preprocessing,Painting,MLModels
from scipy.signal import savgol_filter


data_name = 'w4'
data_path = r"D:\Solar panels\20211110\\" + data_name + "_RT"
img = envi.open(data_path + '.hdr', data_path + '.raw').asarray()
w, h, b = img.shape
img_1d = img.reshape(-1, b)


# %% label

files = glob.glob(r'D:\Solar panels\average\FX10\*.npy')

b_hsi_sample = []
w_hsi_sample = []
l_hsi_sample = []
g_hsi_sample = []
gt = []
all_samples =[]
for i, f in enumerate(files):
    current_file = np.load(f)
    for cf in current_file:
        sample = cf[5:215]
        all_samples.append(sample)
        f = f.split('\\')[-1]
        if 'b' in f:
            b_hsi_sample.append(sample)
            gt.append(0)
        elif 'w' in f:
            w_hsi_sample.append(sample)
            gt.append(1)
        elif 'l' in f:
            l_hsi_sample.append(sample)
            gt.append(2)
        elif 'g' in f:
            g_hsi_sample.append(sample)
            gt.append(3)

state = 789456
b_train,b_test = train_test_split(b_hsi_sample,test_size=0.5,random_state=state)
w_train,w_test = train_test_split(w_hsi_sample,test_size=0.5,random_state=state)
l_train,l_test = train_test_split(l_hsi_sample,test_size=0.5,random_state=state)
g_train,g_test = train_test_split(g_hsi_sample,test_size=0.5,random_state=state)
train_hsi_samples = b_train + w_train + l_train + g_train
test_hsi_samples = b_test + w_test + l_test + g_test
train_gt = [0]*len(b_train) + [1]*len(w_train) +[2]*len(l_train) + [3]*len(g_train)
test_gt = [0]*len(b_test) +[1]*len(w_test) +[2]*len(l_test)+[3]*len(g_test)
# train_hsi_samples,test_hsi_samples,train_gt,test_gt= train_test_split(all_samples,gt,test_size=0.5,random_state=state)
# %% 光譜前處理

train_hs = np.array(train_hsi_samples)
test_hs = np.array(test_hsi_samples)
# train_input = Preprocessing.snv(train_hs)
# test_input = Preprocessing.snv(test_hs)
# train_input = data_normalize(train_hs)
# test_input = data_normalize(test_hs)

# train_sf = savgol_filter(train_hs,13,1)
# test_sf = savgol_filter(test_hs,13,1)
# train_input = snv(train_sf)
# test_input = snv(test_sf)

# train_sf = snv(train_hs)
# test_sf = snv(test_hs)
# train_input = savgol_filter(train_sf,21,1)
# test_input = savgol_filter(test_sf,21,1)

train_input = np.array(train_hsi_samples)
test_input = np.array(test_hsi_samples)
#%% PCA 降維
components = 5
pca = PCA(components)
pca.fit(train_input)
pca_train_input = pca.transform(train_input)
pca_test_input = pca.transform(test_input)
evr = sum(pca.explained_variance_ratio_)
ev = pca.explained_variance_

# 獲取eigen value
eigens = Preprocessing.get_eigen(train_input)

#%% 平均光譜圖

types = ('chip_b','chip_w','line','glass')
Painting.plot_average_spectral(train_input,train_gt,components,types,eigens)
# Painting.plot_average_spectral(train_input,train_gt,components,eigens,types)


#%% train and test

svm = MLModels.SVM_model(train_input,train_gt)
prediction = svm.predict(test_input)
acc = accuracy_score(test_gt,prediction)
cnf_matrix = confusion_matrix(test_gt, prediction)
Painting.plot_confusion_matrix(cnf_matrix, classes=types)



ModelAccess.save_model(pca,'pca'+str(round(acc,3)))
ModelAccess.save_model(svm,'svm_predict'+str(round(acc,3)))