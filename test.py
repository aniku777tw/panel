from spectral.io import envi
from useful_class import ModelAccess,Preprocessing
import matplotlib.pyplot as plt
import numpy as np
import glob

data_name = 'l5'
data_path = r"D:\Solar panels\20211105\\" + data_name + "_RT"
img = envi.open(data_path + '.hdr', data_path + '.raw').asarray()
w, h, b = img.shape
img_1d = img.reshape(-1, b)

bg = ModelAccess.load_model('remove_bg_svm')
pca = ModelAccess.load_model('pca0.912')
svm = ModelAccess.load_model('svm_predict0.912')

mask = bg.predict(img_1d).reshape(w,h)
img_masked = Preprocessing.masking(img,mask)

roi_img=Preprocessing.roi(img_masked,mask,400)
plt.figure()
plt.imshow(mask)
plt.figure()
plt.imshow(img_masked[:,:,101])
plt.figure()
for i in range(1,len(roi_img)+1):
    try:
        plt.subplot(10, 10, i)
        ri = np.stack((roi_img[i-1][...,96],roi_img[i-1][...,58],roi_img[i-1][...,20])).swapaxes(0, 2)
        plt.imshow(ri)
    except:
        pass

# files = glob.glob(r'D:\Solar panels\average\FX10\*.npy')

# b_hsi_sample = []
# w_hsi_sample = []
# l_hsi_sample = []
# g_hsi_sample = []

# for i, f in enumerate(files):
#     current_file = np.load(f)
#     for cf in current_file:
#         sample = cf[5:100]
#         f = f.split('\\')[-1]
#         if 'b' in f:
#             b_hsi_sample.append(sample)
#         elif 'w' in f:
#             w_hsi_sample.append(sample)
#         elif 'l' in f:
#             l_hsi_sample.append(sample)
#         elif 'g' in f:
#             g_hsi_sample.append(sample)


    
# # hsi_mean = Preprocessing.mean(roi_img,b)
# hsi_snv = Preprocessing.snv(np.array(g_hsi_sample))
# hsi_pca = pca.transform(hsi_snv)
# prediction = svm.predict(hsi_pca)

# wrong = np.where(prediction!=3)
# correct = np.where(prediction==3)
# plt.figure()
# for i in wrong[0] :
#     plt.plot(range(b),hsi_snv[i],'r')
# for i in correct[0] :
#     plt.plot(range(b),hsi_snv[i],'b')


