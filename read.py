from scipy import linalg
import glob
from spectral.io import envi
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import itertools
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage import segmentation, measure, morphology, color
from hsipl_tools.preprocessing import snv

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def roi(img, mask, area_value):
    try:
        roi = []
        cleared = mask.copy()
        segmentation.clear_border(cleared)
        label_image = measure.label(cleared)
        borders = np.logical_xor(mask, cleared)
        label_image[borders] = -1
        x = 2
        for region in measure.regionprops(label_image):

            if region.area < area_value:
                continue
            print(np.array(region.bbox))
            minr, minc, maxr, maxc = region.bbox
            roi.append(img[minr-x:maxr+x, minc-x:maxc+x, :])
    except:
        print('ROI error')
    else:
        return roi


def LDA_Model(X_train, y_train):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    model = lda.fit(X_train, y_train)
    return model


def SVM_Model(X_train, y_train):

    from sklearn.svm import SVC
    svm = SVC()
    model = svm.fit(X_train, y_train)
    return model


def SaveModel(model, tag):
    import pickle
    with open('model/' + tag + '.pkl', 'wb') as f:
        pickle.dump(model, f)


def LoadModel(tag):
    import pickle
    with open('model/' + tag + '.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def masking(img, mask):
    new_img = img.copy()
    for i in range(img.shape[2]):
        new_img[..., i] = mask*img[..., i]
    return new_img

def train_masking_model(img):
    plt.imshow(img[:,:,101])
    points = plt.ginput(25,timeout=120)
    points = np.array(points).astype(int)
    plt.close()
    gt=[1]*20 + [0]*5
    train_data = []
    for point in points :
        train_data.append(img[point[1],point[0],:])
    
    # img_1d = snv(img_1d)
    
    model = SVM_Model(train_data,gt)
    ans = model.predict(img_1d)
    plt.imshow(ans.reshape(w,h))
    SaveModel(model,'remove_bg_svm')

def grab_sample(data_name):
    
    data_path = r"D:\Solar panels\20211110\\" + data_name + "_RT"
    img = envi.open(data_path + '.hdr', data_path + '.raw').asarray()
    w, h, b = img.shape
    img_1d = img.reshape(-1, b)
    
    rb_model = LoadModel('remove_bg_svm')
    mask = rb_model.predict(img_1d).reshape(w,h)
    img_masked = masking(img,mask)
    
    roi_img=roi(img_masked,mask,400)
    plt.figure()
    plt.imshow(mask)
    plt.figure()
    plt.imshow(img_masked[:,:,101])
    plt.figure()
    for i in range(1,len(roi_img)+1):
        try:
            plt.subplot(10, 10, i)
            plt.imshow(roi_img[i-1][...,100])
        except:
            pass
    hsi_mean = []
    for i in range(len(roi_img)):
        hsi = roi_img[i]
        hsi_1d = hsi.reshape(-1,b)
        hsi_1d[hsi_1d == 0] = np.nan
        mean = np.nanmean(hsi_1d,axis=0)
        hsi_mean.append(mean)
    
    
    plt.figure()
    for i in range(len(roi_img)):
        plt.plot(range(b),hsi_mean[i])
    np.save('average/FX10/'+data_name+'_average',hsi_mean)

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
        sample = cf
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
train_hsi_gt = [0]*len(b_train) + [1]*len(w_train) +[2]*len(l_train) + [3]*len(g_train)
test_hsi_gt = [0]*len(b_test) +[1]*len(w_test) +[2]*len(l_test)+[3]*len(g_test)
# train_hsi_samples,test_hsi_samples,train_hsi_gt,test_hsi_gt= train_test_split(all_samples,gt,test_size=0.5,random_state=state)
# %% 光譜前處理SNV
train_snv_hs = train_hsi_samples
test_snv_hs = test_hsi_samples

train_hs = np.array(train_hsi_samples)
train_snv_hs = snv(train_hs)
test_hs = np.array(test_hsi_samples)
test_snv_hs = snv(test_hs)



#%% PCA 降維
components = 5
pca = PCA(components)
pca_train_snv_hs = pca.fit_transform(train_snv_hs)
pca_test_snv_hs = pca.transform(test_snv_hs)
evr = sum(pca.explained_variance_ratio_)
ev = pca.explained_variance_

# 獲取eigen value
cov_mat = np.cov(train_snv_hs.T)
eigen_val = linalg.eigvals(cov_mat)
eigen_val_sort = np.argsort(eigen_val)+0




#%% 平均光譜圖

plt.figure()
for i,ths in enumerate(train_snv_hs):
    if train_hsi_gt[i] == 0:
        cb, = plt.plot(range(len(ths)), ths, 'b')
    elif train_hsi_gt[i] == 1:
        cw, = plt.plot(range(len(ths)), ths, 'r')
    elif train_hsi_gt[i] == 2:
        l, = plt.plot(range(len(ths)), ths, 'g')
    elif train_hsi_gt[i] == 3:
        g, = plt.plot(range(len(ths)), ths, 'y')
for i, xc in enumerate(eigen_val_sort[:components]):
    plt.axvline(x=xc)
plt.title('train' + str(eigen_val_sort[:components]))
plt.legend([cb, cw, l, g], ('chip_b', 'chip_w', 'line', 'glass'))

plt.figure()
for i,ths in enumerate(test_snv_hs):
    if test_hsi_gt[i] == 0:
        cb, = plt.plot(range(len(ths)), ths, 'b')
    elif test_hsi_gt[i] == 1:
        cw, = plt.plot(range(len(ths)), ths, 'r')
    elif test_hsi_gt[i] == 2:
        l, = plt.plot(range(len(ths)), ths, 'g')
    elif test_hsi_gt[i] == 3:
        g, = plt.plot(range(len(ths)), ths, 'y')
for i, xc in enumerate(eigen_val_sort[:components]):
    plt.axvline(x=xc)
plt.title('test' + str(eigen_val_sort[:components]))
plt.legend([cb, cw, l, g], ('chip_b', 'chip_w', 'line', 'glass'))


#%% SVM model

svm = SVC()
model = svm.fit(pca_train_snv_hs,train_hsi_gt)


# predict result

prediction = model.predict(pca_test_snv_hs)
acc = accuracy_score(test_hsi_gt,prediction)
stack = np.stack((prediction,test_hsi_gt),axis=1)


plt.figure()
cnf_matrix = confusion_matrix(test_hsi_gt, prediction)
target_names=['chip_b', 'chip_w', 'line', 'glass']
target_names=['chip', 'line', 'glass']
plot_confusion_matrix(cnf_matrix, classes=target_names,normalize=True,
                    title=' confusion matrix')



SaveModel(pca,'pca'+str(round(acc,3)))
SaveModel(model,'svm_predict'+str(round(acc,3)))