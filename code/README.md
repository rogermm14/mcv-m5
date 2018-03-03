# About how to run the code

#### Object recognition

  - VGG 
 
    - Baseline **[TT100K dataset]**
    
    ```
    python train.py -c config/vgg_tt100k_classif.py -e vgg_tt100k_classif 
    ```
    
    - Substract mean and normalize by standard deviation (computed from training set)  **[TT100K dataset]**
    
    ```
    python train.py -c config/vgg_tt100k_classif_preprocessing.py -e vgg_tt100k_classif_preprocessing
    ```
    
    - Substitute the original 224x224 resized images by random crops of 224x224 to feed the net **[TT100K dataset]**
    
    ```
    python train.py -c config/vgg_tt100k_classif_crops.py -e vgg_tt100k_classif_crops
    ```
    
    - Transfer learning to another dataset **[TT100K dataset -> BelgiumTS dataset]**
    
    ```
    python train.py -c config/vgg_belgiumTS_classif_transfer.py -e vgg_belgiumTS_classif_transfer
    ```
    
    - Baseline **[KITTI dataset]**
    
    ```
    python train.py -c config/vgg_kitti_classif.py -e vgg_tt100k_classif
    ``` 
   
    - Fine-tuning based on ImageNet weights  **[KITTI dataset]**
    
    ```
    python train.py -c config/vgg_kitti_classif_finetune.py -e vgg_tt100k_classif_finetune
    ``` 

  - ResNet

    - Baseline **[TT100K dataset]**
    
    ```
    python train.py -c config/resnet_tt100k_classif.py -e resnet_tt100k_classif
    ```  

    - Fine-tuning based on ImageNet weights **[TT100K dataset]**
 
    ```
    python train.py -c config/resnet_tt100k_classif_finetune.py -e resnet_tt100k_classif_finetune
    ```  
    
    - Boost performance (Data augmentation) **[TT100K dataset]**
 
    ```
    python train.py -c config/resnet_tt100k_classif_opt.py -e resnet_tt100k_classif_dataaug
    ```  
    
    - Boost performance (Data augmentation + Drop-out layers) **[TT100K dataset]**
 
    ```
    python train.py -c config/resnet_tt100k_classif_dataaug_dropout.py -e resnet_tt100k_classif_dataaug_dropout
    ```  
    
    - Boost performance (Data augmentation + Drop-out layers + Optimized Paramaters) **[TT100K dataset]**
 
    ```
    python train.py -c config/resnet_tt100k_classif_dataaug_dropout_opt.py -e resnet_tt100k_classif_dataaug_dropout_opt
    ``` 
   
