# About how to run the code

## Weeks 1-2. Object recognition

To run all experiments:  
    ```
    cd ../other_scripts  
    ```    
    ```
    bash run_experiments_week2.sh
    ```

- **Experiments with VGG-16:**    
 
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
    python train.py -c config/vgg_kitti_classif.py -e vgg_kitti_classif
    ``` 
   
    - Fine-tuning based on ImageNet weights  **[KITTI dataset]**
    ```
    python train.py -c config/vgg_kitti_classif_finetune.py -e vgg_kitti_classif_finetune
    ``` 

 - **Experiments with ResNet-50:**   

    - Baseline **[TT100K dataset]**
    ```
    python train.py -c config/resnet_tt100k_classif.py -e resnet_tt100k_classif
    ```  

    - Fine-tuning based on ImageNet weights **[TT100K dataset]**
    ```
    python train.py -c config/resnet_tt100k_classif_finetune.py -e resnet_tt100k_classif_finetune
    ```  
    
    - Retrain all layers using ImageNet weights as initialization **[TT100K dataset]**
    ```
    python train.py -c config/resnet_tt100k_classif_dataaug.py -e resnet_tt100k_classif_dataaug
    ```   
    
    - Boost performance experiment 1 (Data augmentation) **[TT100K dataset]**
    ```
    python train.py -c config/resnet_tt100k_classif_opt.py -e resnet_tt100k_classif_dataaug
    ```  
    
    - Boost performance experiment 2 (resnet50_v2) **[TT100K dataset]**
    ```
    python train.py -c config/resnet_tt100k_classif_resnet50v2.py -e resnet_tt100k_classif_resnet50v2
    ```  

## Weeks 3-4. Object detection

To run all experiments:  
    ```
    cd ../other_scripts  
    ```    
    ```
    bash run_experiments_week3.sh
    ```

- **Experiments with YOLO:**    
 
    - Baseline **[TT100K dataset]**
    ```
    python train.py -c config/yolo_tt100k_detection.py -e yolo_tt100k_detection 
    ```
    
    - Baseline  **[Udacity dataset]**
    ```
    python train.py -c config/yolo_Udacity_detection.py -e yolo_Udacity_detection  
    ```    

    - Baseline + data augmentation **[TT100K dataset]**
    ```
    python train.py -c config/yolo_dataaug_tt100k_detection.py -e yolo_dataaug_tt100k_detection
    ```
    
    - Baseline + data augmentation  **[Udacity dataset]**
    ```
    python train.py -c config/yolo_dataaug_Udacity_detection.py -e yolo_dataaug_Udacity_detection  
    ```   
    
- **Experiments with Faster R-CNN:**  

    - Baseline **[TT100K dataset]**
    ```
    cd ../other_scripts  
    python create_gt_frcnn_TT100K_detection.py
    cd ../frcnn
    python train_frcnn.py -o simple -p ../other_scripts/TT100K_detection_train_labels.txt
    python test_frcnn.py -p ../Datasets/detection/TT100K_detection/test
    cd ../other_scripts
    python evaluate_frcnn_TT100K_detection.py

    ```  
    ```  
    rm -r ../frcnn/predictions
    rm ../frcnn/model_frcnn.hdf5
    rm ../frcnn/config.pickle
    mkdir ../frcnn/predictions
    ```  

    - Baseline **[Udacity dataset]**
    ```  
    python create_gt_frcnn_Udacity.py
    cd ../frcnn
    python train_frcnn.py -o simple -p ../other_scripts/Udacity_train_labels.txt
    python test_frcnn.py -p ../Datasets/detection/Udacity/test
    cd ../other_scripts
    python evaluate_frcnn_Udacity.py
    ```
    
