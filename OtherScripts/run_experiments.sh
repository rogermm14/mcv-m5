# !/bin/bash

# Execute experiment's list

# Baseline [TT100K dataset]

# ---------- VGG ------------

python train.py -c config/vgg_tt100k_classif.py -e vgg_default   

# Substract mean and normalize by standard deviation (computed from training set) [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_preprocessing.py -e vgg_preprocessing  

# Use random crops of 224x224 to feed the net [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_crops.py -e vgg_input_crops    

# Transfer learning to another dataset [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_resize.py -e vgg_input_resize  

# Transfer learning to another dataset [TT100K dataset -> BelgiumTS dataset]

python train.py -c config/vgg_belgiumTS_classif_transfer.py -e vgg_belgiumTS_classif_transfer

# Baseline [KITTI dataset]

python train.py -c config/vgg_kitti_classif_baseline.py -e vgg_tt100k_classif_baseline

# Fine-tuning based on ImageNet weights [KITTI dataset]

python train.py -c config/vgg_kitti_classif_finetune.py -e vgg_tt100k_classif_finetune

# ---------- ResNet ------------

# Baseline [TT100K dataset]

python train.py -c config/resnet_tt100k_classif_baseline.py -e resnet_tt100k_classif_baseline

# Fine-tuning based on ImageNet weights [TT100K dataset]

# python train.py -c config/resnet_tt100k_classif_finetune.py -e resnet_tt100k_classif_finetune

