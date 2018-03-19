import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import errno


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

path_train = '../Datasets/detection/'+dataset_name+'/train/*.txt'
path_valid = '../Datasets/detection/'+dataset_name+'/valid/*.txt'
path_test = '../Datasets/detection/'+dataset_name+'/test/*.txt'

#Number of images per split
train_images = np.size(os.listdir('../Datasets/detection/'+dataset_name+'/train/'))
print 'Number of train images:',train_images/2
valid_images = np.size(os.listdir('../Datasets/detection/'+dataset_name+'/valid/'))
print 'Number of validation images:',valid_images/2
test_images  = np.size(os.listdir('../Datasets/detection/'+dataset_name+'/test/'))
print 'Number of test images:',test_images/2

#---READ .txt FILES---
train_vec = np.array([])
valid_vec = np.array([])
test_vec = np.array([])
#Train
files = glob.glob(path_train)
for name in files:
    try:
        with open(name) as f:
            for line in f.readlines():
                if(line[1]==' '):
                    train_vec = np.append(train_vec, line[0])
                else:
                    train_vec = np.append(train_vec, line[0]+line[1])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
#Validation
files = glob.glob(path_valid)
for name in files:
    try:
        with open(name) as f:
            for line in f.readlines():
                if(line[1]==' '):
                    valid_vec = np.append(valid_vec, line[0])
                else:
                    valid_vec = np.append(valid_vec, line[0]+line[1])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
#Test
files = glob.glob(path_test)
for name in files:
    try:
        with open(name) as f:
            for line in f.readlines():
                if(line[1]==' '):
                    test_vec = np.append(test_vec, line[0])
                else:
                    test_vec = np.append(test_vec, line[0]+line[1])
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

#---COUNT INSTANCES OF EACH CLASS---
#Train
train_vec = train_vec.tolist()
train_vec_final = np.zeros(len(classes))
for c in range(0,len(classes)):
    train_vec_final[c] = train_vec.count(str(c))

#Validation
valid_vec = valid_vec.tolist()
valid_vec_final = np.zeros(len(classes))
for c in range(0,len(classes)):
    valid_vec_final[c] = valid_vec.count(str(c))

#Test
test_vec = test_vec.tolist()
test_vec_final = np.zeros(len(classes))
for c in range(0,len(classes)):
    test_vec_final[c] = test_vec.count(str(c))

#---PLOT HISTOGRAMS---
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(train_vec_final)),train_vec_final)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_train_TT100K_detection.png')
plt.close(fig)
     
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(valid_vec_final)),valid_vec_final)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_valid_TT100K_detection.png')
plt.close(fig)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(test_vec_final)),test_vec_final)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_test_TT100K_detection.png')
plt.close(fig)
