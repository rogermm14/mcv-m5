# !/bin/bash

# Execute experiments list

cd ..
cd code

# ---------- VGG ------------

#### Baseline code [TT100K dataset]

python train.py -c config/vgg_tt100k_classif.py -e vgg_tt100k_classif

#### Substract mean and normalize by standard deviation (computed from training set) [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_preprocessing.py -e vgg_tt100k_classif_preprocessing  

#### Use random crops of 224x224 to feed the net [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_crops.py -e vgg_tt100k_classif_crops

#### Transfer learning to another dataset [TT100K dataset -> BelgiumTS dataset]

python train.py -c config/vgg_belgiumTS_classif_transfer.py -e vgg_belgiumTS_classif_transfer

#### Baseline code [KITTI dataset]

python train.py -c config/vgg_kitti_classif.py -e vgg_kitti_classif

#### Fine-tuning based on ImageNet weights [KITTI dataset]

#python train.py -c config/vgg_kitti_classif_finetune.py -e vgg_kitti_classif_finetune

# ---------- ResNet ------------

#### Baseline [TT100K dataset]

python train.py -c config/resnet_tt100k_classif.py -e resnet_tt100k_classif

#### Fine-tuning based on ImageNet weights [TT100K dataset]

python train.py -c config/resnet_tt100k_classif_finetune.py -e resnet_tt100k_classif_finetune

#### Retrain all layers using ImageNet weights as initialization [TT100K dataset]

python train.py -c config/resnet_tt100k_classif_finetune2.py -e resnet_tt100k_classif_finetune2

#### Boost performance experiment 1 (data augmentation)

python train.py -c config/resnet_tt100k_classif_dataaug.py -e resnet_tt100k_classif_dataaug

#### Boost performance experiment 2 (resnet50_v2)

python train.py -c config/resnet_tt100k_classif_resnet50v2.py -e resnet_tt100k_classif_resnet50v2

