# !/bin/bash

# Execute experiments list

cd ..
cd code

# ---------- FCN8 ------------

#### Baseline code [CamVid dataset]

python train.py -c config/fcn8_camvid_segmentation.py -e fcn8_camvid_segmentation

#### Baseline code [KITTI segmentation dataset]

python train.py -c config/fcn8_kitti_segmentation.py -e fcn8_kitti_segmentation


# ---------- SEGNET ------------

#### SegNet-VGG16 trained with baseline code [CamVid dataset]

python train.py -c config/segnet-vgg16_camvid_segmentation.py -e segnet-vgg16_camvid_segmentation

#### SegNet-Basic trained with baseline code [CamVid dataset]

python train.py -c config/segnet-basic_camvid_segmentation.py -e segnet-basic_camvid_segmentation

#### SegNet-VGG16 trained with custom strategy [KITTI segmentation dataset]

python train.py -c config/segnet-vgg16_kitti_segmentation.py -e segnet-vgg16_kitti_segmentation


# ---------- UNET ------------

# To be completed
