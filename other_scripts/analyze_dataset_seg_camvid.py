import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import errno
import cv2

dataset_name = 'camvid'
nclasses = 12
classes = {
        0: 'sky',
        1: 'building',
        2: 'column_pole',
        3: 'road',
        4: 'sidewalk',
        5: 'tree',
        6: 'sign',
        7: 'fence',
        8: 'car',
        9: 'pedestrian',
        10: 'byciclist',
        11: 'void'
}



path_train = '../Datasets/segmentation/'+dataset_name+'/train/masks/*.png'
path_valid = '../Datasets/segmentation/'+dataset_name+'/valid/masks/*.png'
path_test = '../Datasets/segmentation/'+dataset_name+'/test/masks/*.png'

#Number of images per split
train_images = np.size(os.listdir('../Datasets/segmentation/'+dataset_name+'/train/images/'))
print 'Number of train images:',train_images
valid_images = np.size(os.listdir('../Datasets/segmentation/'+dataset_name+'/valid/images/'))
print 'Number of validation images:',valid_images
test_images  = np.size(os.listdir('../Datasets/segmentation/'+dataset_name+'/test/images/'))
print 'Number of test images:',test_images

#Train
pixels_per_class = np.zeros(nclasses)
files = glob.glob(path_train)
for name in files:
    current_mask = cv2.imread(name,-1)
    hist, bin_edges = np.histogram(current_mask, bins=np.arange(nclasses+1)-0.5, density=False)
    pixels_per_class = pixels_per_class + hist
norm_pixels_per_class_train = pixels_per_class.astype(float)/sum(pixels_per_class)

#Validation
pixels_per_class = np.zeros(nclasses)
files = glob.glob(path_valid)
for name in files:
    current_mask = cv2.imread(name,-1)
    hist, bin_edges = np.histogram(current_mask, bins=np.arange(nclasses+1)-0.5, density=False)
    pixels_per_class = pixels_per_class + hist
norm_pixels_per_class_val = pixels_per_class.astype(float)/sum(pixels_per_class)

#Test
pixels_per_class = np.zeros(nclasses)
files = glob.glob(path_test)
for name in files:
    current_mask = cv2.imread(name,-1)
    hist, bin_edges = np.histogram(current_mask, bins=np.arange(nclasses+1)-0.5, density=False)
    pixels_per_class = pixels_per_class + hist
norm_pixels_per_class_test = pixels_per_class.astype(float)/sum(pixels_per_class)

#---PLOT HISTOGRAMS---
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(norm_pixels_per_class_train)),norm_pixels_per_class_train)
ax.set_xlabel('class index')
ax.set_ylabel('pixels percentage')
fig.savefig('hist_train_camvid.png')
plt.close(fig)
     
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(norm_pixels_per_class_val)),norm_pixels_per_class_val)
ax.set_xlabel('class index')
ax.set_ylabel('pixels percentage')
fig.savefig('hist_valid_camvid.png')
plt.close(fig)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(norm_pixels_per_class_test)),norm_pixels_per_class_test)
ax.set_xlabel('class index')
ax.set_ylabel('pixels percentage')
fig.savefig('hist_test_camvid.png')
plt.close(fig)
