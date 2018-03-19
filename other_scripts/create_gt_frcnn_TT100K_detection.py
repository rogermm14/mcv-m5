import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import errno
import re
import cv2

dataset_name = 'TT100K_detection'

classes = {0:'i2',
           1:'i4',
           2:'i5',
           3:'il100',
           4:'il60',
           5:'il80',
           6:'io',
           7:'ip',
           8:'p10',
           9:'p11',
           10:'p12',
           11:'p19',
           12:'p23',
           13:'p26',
           14:'p27',
           15:'p3',
           16:'p5',
           17:'p6',
           18:'pg',
           19:'ph4',
           20:'ph4.5',
           21:'ph5',
           22:'pl100',
           23:'pl120',
           24:'pl20',
           25:'pl30',
           26:'pl40',
           27:'pl5',
           28:'pl50',
           29:'pl60',
           30:'pl70',
           31:'pl80',
           32:'pm20',
           33:'pm30',
           34:'pm55',
           35:'pn',
           36:'pne',
           37:'po',
           38:'pr40',
           39:'w13',
           40:'w32',
           41:'w55',
           42:'w57',
           43:'w59',
           44:'wo'}

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
                if (line[1] == ' '):
                    class_id = line[0]
                else:
                    class_id = line[0] + line[1]
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
                if (line[1] == ' '):
                    class_id = line[0]
                else:
                    class_id = line[0] + line[1]
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
                if (line[1] == ' '):
                    class_id = line[0]
                else:
                    class_id = line[0] + line[1]
            	file.write(os.path.splitext(name)[0]+'.jpg,'+str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+classes[int(class_id)]+'\n')
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
