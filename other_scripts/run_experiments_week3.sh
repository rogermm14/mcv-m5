# !/bin/bash

# Execute experiment's list

# Baseline [TT100K dataset]

cd ..
cd code

# ---------- YOLO ------------

#### Baseline code [TT100K dataset]

#python train.py -c config/yolo_tt100k_detection.py -e yolo_tt100k_detection

#### Baseline [Udacity dataset]

python train.py -c config/yolo_Udacity_detection.py -e yolo_Udacity_detection  

#### Data augmentation [TT100K dataset]

python train.py -c config/yolo_dataaug_tt100k_detection.py -e yolo_dataaug_tt100k_detection

#### Data augmentation [Udacity dataset]

python train.py -c config/yolo_dataaug_Udacity_detection.py -e yolo_dataaug_Udacity_detection
