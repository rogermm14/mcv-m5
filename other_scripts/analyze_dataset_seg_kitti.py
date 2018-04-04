import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import errno
import cv2

dataset_name = 'kitti'
nclasses = 12
color_map = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)          # Void
}



path_train = '../Datasets/segmentation/'+dataset_name+'/train/masks/*.png'
path_valid = '../Datasets/segmentation/'+dataset_name+'/valid/masks/*.png'

#Number of images per split
train_images = np.size(os.listdir('../Datasets/segmentation/'+dataset_name+'/train/images/'))
print 'Number of train images:',train_images
valid_images = np.size(os.listdir('../Datasets/segmentation/'+dataset_name+'/valid/images/'))
print 'Number of validation images:',valid_images

#Train
pixels_per_class = np.zeros(nclasses)
files = glob.glob(path_train)
count = 0
for name in files:
    count = count + 1
    print count
    current_mask = cv2.imread(name,-1)
    hist, bin_edges = np.histogram(current_mask, bins=np.arange(nclasses+1)-0.5, density=False)
    pixels_per_class = pixels_per_class + hist
    print pixels_per_class
norm_pixels_per_class_train = pixels_per_class.astype(float)/sum(pixels_per_class)

#Validation
pixels_per_class = np.zeros(nclasses)
files = glob.glob(path_valid)
for name in files:
    current_mask = cv2.imread(name,-1)
    hist, bin_edges = np.histogram(current_mask, bins=np.arange(nclasses+1)-0.5, density=False)
    pixels_per_class = pixels_per_class + hist
norm_pixels_per_class_val = pixels_per_class.astype(float)/sum(pixels_per_class)

#---PLOT HISTOGRAMS---
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(norm_pixels_per_class_train)),norm_pixels_per_class_train)
ax.set_xlabel('class index')
ax.set_ylabel('pixels percentage')
fig.savefig('hist_train_kitti.png')
plt.close(fig)
     
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(norm_pixels_per_class_val)),norm_pixels_per_class_val)
ax.set_xlabel('class index')
ax.set_ylabel('pixels percentage')
fig.savefig('hist_valid_kitti.png')
plt.close(fig)

