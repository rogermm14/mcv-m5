import os
import numpy as np
import matplotlib.pyplot as plt


dataset_name = 'TT100K_trafficSigns'

classes = ('i2','i4','i5','il100','il60','il80','io','ip','p10',
           'p11','p12','p19','p23','p26','p27','p3','p5','p6','pg',
           'ph4','ph4.5','ph5','pl100','pl120','pl20','pl30','pl40',
           'pl5','pl50','pl60','pl70','pl80','pm20','pm30','pm55','pn',
           'pne','po','pr40','w13','w32','w55','w57','w59','wo')

path_train = '../Datasets/classification/'+dataset_name+'/train/'
path_valid = '../Datasets/classification/'+dataset_name+'/valid/'
path_test = '../Datasets/classification/'+dataset_name+'/test/'

train_vec = np.array([])
valid_vec = np.array([])
test_vec = np.array([])

#for fold in train_fold:
for c in range(0,len(classes)):
    train_images = np.size(os.listdir(os.path.join(path_train,classes[c])))
    valid_images = np.size(os.listdir(os.path.join(path_valid,classes[c])))
    test_images  = np.size(os.listdir(os.path.join(path_test, classes[c])))
    train_vec = np.append(train_vec, train_images)
    valid_vec = np.append(valid_vec, valid_images)
    test_vec  = np.append(test_vec, test_images)
    

fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(train_vec)),train_vec)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_train.png')
plt.close(fig)
     
fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(valid_vec)),valid_vec)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_valid.png')
plt.close(fig)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(test_vec)),test_vec)
ax.set_xlabel('class index')
ax.set_ylabel('images')
fig.savefig('hist_test.png')
plt.close(fig)
