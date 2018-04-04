import os
import glob
import numpy as np


# based on model.test() from M5 - Visual Recognition framework
# computes precison, recall and fscore given the detections of the faster r-cnn

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
	self.w, self.h = float(), float()
	self.c = float()
	self.class_num = classes
	self.probs = np.zeros((classes,))

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def main():

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
    keys = classes.keys()
    values = classes.values()

    detection_threshold = 0.3
    total_true = 0.
    total_pred = 0.
    ok = 0.

    files = glob.glob('../frcnn/predictions/*.txt')
    for name in files:
	
        boxes_pred = []
	with open(name) as f:
	    for line in f.readlines():
	        tmp = line.split()
		x, y, w, h = float(tmp[0]), float(tmp[1]), float(tmp[2]), float(tmp[3])
		label = keys[values.index(tmp[4])]
		bx = BoundBox(len(classes))
		bx.probs[int(label)] = 1.
		bx.x, bx.y, bx.w, bx.h = x, y, w, h
		boxes_pred.append(bx)

        boxes_true = []
        label_path = '../Datasets/detection/'+dataset_name+'/test/'+os.path.basename(name)
	gt = np.loadtxt(label_path)

        if len(gt.shape) > 1:
	    for j in range(gt.shape[0]):
	        bx = BoundBox(len(classes))
	        bx.probs[int(gt[j,0])] = 1.
	        bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
	        boxes_true.append(bx)

        total_true += len(boxes_true)
	true_matched = np.zeros(len(boxes_true))
	for b in boxes_pred:
	    if b.probs[np.argmax(b.probs)] < detection_threshold:
                continue
	    total_pred += 1.
	    for t,a in enumerate(boxes_true):
                if true_matched[t]:
		    continue
		if box_iou(a, b) > 0.5 and np.argmax(a.probs) == np.argmax(b.probs):
		    true_matched[t] = 1
		    ok += 1.
		    break
	
    print 'total_true:',total_true,' total_pred:',total_pred,' ok:',ok
    p = 0. if total_pred == 0 else (ok/total_pred)
    r = ok/total_true
    print('Precission = ' + str(p))
    print('Recall     = ' + str(r))
    f = 0. if (p + r) == 0 else (2*p*r/(p + r))
    print('F-score    = '+str(f))

if __name__ == '__main__': main()