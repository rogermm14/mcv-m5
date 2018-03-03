# !/bin/bash

# Execute experiment's list

# Baseline [TT100K dataset]

python train.py -c config/vgg_tt100k_classif.py -e vgg_default   

# Substract mean and normalize by standard deviation (computed from training set) [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_preprocessing.py -e vgg_preprocessing  

# Use random crops of 224x224 to feed the net [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_crops.py -e vgg_input_crops    

# Transfer learning to another dataset [TT100K dataset]

python train.py -c config/vgg_tt100k_classif_resize.py -e vgg_input_resize  
