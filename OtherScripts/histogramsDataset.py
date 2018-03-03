import os
import numpy as np
import matplotlib.pyplot as plt

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

path_train = '../Datasets/classification/TT100K_trafficSigns/train/'
path_test = '../Datasets/classification/TT100K_trafficSigns/test/'

train_fold =  os.listdir(path_train)
test_fold = os.listdir(path_test)
train_vec= np.array([])


#for fold in train_fold:
for c in range(0,len(classes)):

    items=np.size(os.listdir(os.path.join(path_train,classes[c])))
    train_vec = np.append(train_vec,items)
    
test_vec= np.array([])
#for fold in test_fold:
for c in range(0,len(classes)):

    items=np.size(os.listdir(os.path.join(path_test,classes[c])))
    test_vec = np.append(test_vec,items)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.bar(range(0,len(train_vec)),train_vec)
ax.set_ylabel('images')
ax.set_xlabel('class')
fig.savefig('hist_train.png')
plt.close()
     
fig2=plt.figure()
ax2=fig2.add_subplot(111)
ax2.bar(range(0,len(test_vec)),test_vec)
ax2.set_ylabel('images2')
ax2.set_xlabel('class2')
fig.savefig('hist_test.png')
plt.close()
