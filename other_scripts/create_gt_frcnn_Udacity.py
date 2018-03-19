import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import errno
import re
import cv2


dataset_name = 'Udacity'

classes = {0:'Car',
           1:'Pedestrian',
           2:'Truck'}

path_train = '../Datasets/detection/' + dataset_name + '/train'
path_valid = '../Datasets/detection/' + dataset_name + '/valid'
path_test = '../Datasets/detection/' + dataset_name + '/test'

# ---READ .txt FILES---
train_vec = np.array([])
valid_vec = np.array([])
test_vec = np.array([])
# Train
files = glob.glob(path_train + '/*.txt')
file = open(dataset_name+'_train_labels.txt','w') 
for name in files:
    try:
        with open(name) as f:
	    frame = cv2.imread(os.path.splitext(name)[0]+'.jpg', 0)
	    h, w = frame.shape
            for line in f.readlines():
                bbox_coord = re.findall("\d+\.\d+", line)
                x = float(bbox_coord[0])
                y = float(bbox_coord[1])
                wb = float(bbox_coord[2])
                hb = float(bbox_coord[3])
                x1 = int (round((x - wb/2.) * w))
                y1 = int (round((y - hb/2.) * h))
                x2 = int (round((x + wb/2.) * w))
                y2 = int (round((y + hb/2.) * h))
                #print str(x1)+','+str(y1)+','+ str(x2)+','+str(y2)
                if x1 < 0 :  x1 = 0
                if x2 > w - 1 : x2 = w - 1
                if y1 < 0 :   y1 = 0
                if y2 > h - 1 : y2 = h - 1
                class_id = line[0]
            	file.write(os.path.splitext(name)[0]+'.jpg,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classes[int(class_id)]+'\n')
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
file.close() 

# Validation
files = glob.glob(path_valid + '/*.txt')
file = open(dataset_name+'_valid_labels.txt','w') 
for name in files:
    try:
        with open(name) as f:
	    frame = cv2.imread(os.path.splitext(name)[0]+'.jpg', 0)
	    h, w = frame.shape
            for line in f.readlines():
            	bbox_coord = re.findall("\d+\.\d+", line)
                x = float(bbox_coord[0])
                y = float(bbox_coord[1])
                wb = float(bbox_coord[2])
                hb = float(bbox_coord[3])
                x1 = int (round((x - wb/2.) * w))
                y1 = int (round((y - hb/2.) * h))
                x2 = int (round((x + wb/2.) * w))
                y2 = int (round((y + hb/2.) * h))
                if x1 < 0 :  x1 = 0
                if x2 > w - 1 : x2 = w - 1
                if y1 < 0 :   y1 = 0
                if y2 > h - 1 : y2 = h - 1
                class_id = line[0]
            	file.write(os.path.splitext(name)[0]+'.jpg,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classes[int(class_id)]+'\n')
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
file.close() 

# Test
files = glob.glob(path_test + '/*.txt')
file = open(dataset_name+'_test_labels.txt','w') 
for name in files:
    try:
        with open(name) as f:
	    frame = cv2.imread(os.path.splitext(name)[0]+'.jpg', 0)
	    h, w = frame.shape
            for line in f.readlines():
                bbox_coord = re.findall("\d+\.\d+", line)
                x = float(bbox_coord[0])
                y = float(bbox_coord[1])
                wb = float(bbox_coord[2])
                hb = float(bbox_coord[3])
                x1 = int (round((x - wb/2.) * w))
                y1 = int (round((y - hb/2.) * h))
                x2 = int (round((x + wb/2.) * w))
                y2 = int (round((y + hb/2.) * h))
                if x1 < 0 :  x1 = 0
                if x2 > w - 1 : x2 = w - 1
                if y1 < 0 :   y1 = 0
                if y2 > h - 1 : y2 = h - 1
                class_id = line[0]
            	file.write(os.path.splitext(name)[0]+'.jpg,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classes[int(class_id)]+'\n')
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
