from scipy import linalg
from spectral.io import envi
import matplotlib.pyplot as plt
import numpy as np
import itertools
from skimage import segmentation, measure,transform


class DataLoading:
    
    def grab_sample_cube(data_name,data_path,save_path):
        # data_name = 'b2'
        # data_path = r'D:\Solar panels\20211110\\'
        data_path = data_path + data_name + "_RT"
        
        img = envi.open(data_path + '.hdr', data_path + '.raw').asarray()
        w, h, b = img.shape
        img_1d = img.reshape(-1, b)
        
        rb_model = ModelAccess.load_model('remove_bg_svm')
        mask = rb_model.predict(img_1d).reshape(w,h)
        img_masked = Preprocessing.masking(img,mask)
        
        roi_img = Preprocessing.roi(img_masked,mask,400)
        

        cube_data = []
        
        for i in roi_img:
            resize_data = transform.resize(i,(87,185))
            cube = np.reshape(resize_data,[87,185,b])
            cube_data.append(cube)
        
        
        plt.figure()
        for i in range(len(cube_data)):
            plt.subplot(int(pow(len(cube_data),1/2)) + 1,int(pow(len(cube_data),1/2)) + 1,i+1)
            plt.imshow(cube_data[i][:,:,1])
        all_data = np.array(cube_data)
        # save_path = r"D:\Solar panels\roi\\"+ data_name
        save_path = save_path + data_name
        np.save(save_path ,all_data)  
    
    def grab_sample_average(data_name,data_path,save_path):
        # DataLoading.grab_sample('g1',"D:\Solar panels\\20211110\\","D:\Solar panels\\average\\FX10\\")
        data_path = data_path + data_name + "_RT"
        img = envi.open(data_path + '.hdr', data_path + '.raw').asarray()
        w, h, b = img.shape
        img_1d = img.reshape(-1, b)
        
        rb_model = ModelAccess.load_model('remove_bg_svm')
        mask = rb_model.predict(img_1d).reshape(w,h)
        img_masked = Preprocessing.masking(img,mask)
        
        roi_img = Preprocessing.roi(img_masked,mask,400)
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
        hsi_mean = Preprocessing.mean(roi_img,b)
        plt.figure()
        for i in range(len(roi_img)):
            plt.plot(range(b),hsi_mean[i])
        np.save(save_path + data_name+'_average',hsi_mean)


class Preprocessing:
    
    def get_eigen(marrix_before_pca):
        cov_mat = np.cov(marrix_before_pca.T)
        eigen_val = linalg.eigvals(cov_mat)
        eigen_val_sort = np.argsort(eigen_val)+0
        return eigen_val_sort
    
    def masking(img, mask):
        new_img = img.copy()
        for i in range(img.shape[2]):
            new_img[..., i] = mask*img[..., i]
        return new_img
    
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
        
        
    def mean(roi_img,bands):
        hsi_mean = []

        for i in range(len(roi_img)):
            hsi = roi_img[i]
            hsi_1d = hsi.reshape(-1,bands)
            hsi_1d[hsi_1d == 0] = np.nan
            mean = np.nanmean(hsi_1d,axis=0)
            hsi_mean.append(mean)
        return hsi_mean
    
    def snv(input_data):
        '''
        input_data : hyperspectral data can be
                    3d [width , height , bands] 
                    2d [average spectral samples, bands]
        '''
        d = len(input_data.shape)-1
        mean = input_data.mean(d)
        std = input_data.std(d)
        res = (input_data - mean[...,None] ) / std[...,None]
        return res
    def msc(input_data, reference=None):
        """
            :msc: Scatter Correction technique performed with mean of the sample data as the reference.
            :param input_data: Array of spectral data
            :type input_data: DataFrame
            :returns: data_msc (ndarray): Scatter corrected spectra data
        """
        eps = np.finfo(np.float32).eps
        input_data = np.array(input_data, dtype=np.float64)
        ref = []
        sampleCount = int(len(input_data))
    
        # mean centre correction
        for i in range(input_data.shape[0]):
            input_data[i,:] -= input_data[i,:].mean()
        
        # Get the reference spectrum. If not given, estimate it from the mean
        # Define a new array and populate it with the corrected data    
        data_msc = np.zeros_like(input_data)
        for i in range(input_data.shape[0]):
            for j in range(0, sampleCount, 10):
                ref.append(np.mean(input_data[j:j+10], axis=0))
                # Run regression
                fit = np.polyfit(ref[i], input_data[i,:], 1, full=True)
                # Apply correction
                data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0]
        
        return (data_msc)
    
class ModelAccess:
    
    def load_model(tag):
        import pickle
        with open('model/' + tag + '.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    
    def save_model(model, tag):
        import pickle
        with open('model/' + tag + '.pkl', 'wb') as f:
            pickle.dump(model, f)


class MLModels:
    
    def LDA_model(X_train, y_train):

        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis()
        model = lda.fit(X_train, y_train)
        return model


    def SVM_model(X_train, y_train):

        from sklearn.svm import SVC
        svm = SVC()
        model = svm.fit(X_train, y_train)
        return model
    
    def train_masking_model(img):
        w, h, b = img.shape
        img_1d = img.reshape(-1, b)
        plt.imshow(img[:,:,101])
        points = plt.ginput(25,timeout=120)
        points = np.array(points).astype(int)
        plt.close()
        gt=[1]*20 + [0]*5
        train_data = []
        for point in points :
            train_data.append(img[point[1],point[0],:])
        
        
        model = MLModels.SVM_Model(train_data,gt)
        ans = model.predict(img_1d)
        plt.imshow(ans.reshape(w,h))
        ModelAccess.saveModel(model,'remove_bg_svm')
    

class Painting:
    
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
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
    
    
    def plot_average_spectral(spectral_arrays,gt_arrays,components,types,eigens):
        plt.figure()
        for i,sa in enumerate(spectral_arrays):
            if gt_arrays[i] == 0:
                cb, = plt.plot(range(len(sa)), sa, 'b',alpha=0.5)
            elif gt_arrays[i] == 1:
                cw, = plt.plot(range(len(sa)), sa, 'r',alpha=0.5)
            elif gt_arrays[i] == 2:
                l, = plt.plot(range(len(sa)), sa, 'g',alpha=0.5)
            elif gt_arrays[i] == 3:
                g, = plt.plot(range(len(sa)), sa, 'y',alpha=0.5)
        
        plt.legend([cb, cw, l, g], types)     
        

        for i, xc in enumerate(eigens[:components]):
            plt.axvline(x=xc)
            plt.title('train' + str(eigens[:components]))    
