# !/bin/bash

# Execute experiment's list

# Baseline [TT100K dataset]

cd ..
cd code

# ---------- YOLO ------------

#### Baseline code [TT100K dataset]

python train.py -c config/yolo_tt100k_detection.py -e yolo_tt100k_detection

#### Baseline [Udacity dataset]

python train.py -c config/yolo_Udacity_detection.py -e yolo_Udacity_detection  

#### Data augmentation [TT100K dataset]

python train.py -c config/yolo_dataaug_tt100k_detection.py -e yolo_dataaug_tt100k_detection

#### Data augmentation [Udacity dataset]

python train.py -c config/yolo_dataaug_Udacity_detection.py -e yolo_dataaug_Udacity_detection

# ---------- Faster R-CNN ------------ 

#### Baseline code [TT100K dataset]

cd ../other_scripts
python create_gt_frcnn_TT100K_detection.py
cd ../frcnn
python train_frcnn.py -o simple -p ../other_scripts/TT100K_detection_train_labels.txt
python test_frcnn.py -p ../Datasets/detection/TT100K_detection/test
cd ../other_scripts
python evaluate_frcnn_TT100K_detection.py


rm -r ../frcnn/predictions
rm ../frcnn/model_frcnn.hdf5
rm ../frcnn/config.pickle
mkdir ../frcnn/predictions


#### Baseline code [Udacity dataset]

python create_gt_frcnn_Udacity.py
cd ../frcnn
python train_frcnn.py -o simple -p ../other_scripts/Udacity_train_labels.txt
python test_frcnn.py -p ../Datasets/detection/Udacity/test
cd ../other_scripts
python evaluate_frcnn_Udacity.py
 
